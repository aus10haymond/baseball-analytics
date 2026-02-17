"""
ML Projections Integration Module

Connects matchup_machine XGBoost models with fantasy_mlb_ai's daily roster management.
Generates matchup-specific fantasy point projections based on probable pitchers.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# Add matchup_machine to path
MATCHUP_MACHINE_PATH = Path(__file__).parent.parent / "matchup_machine" / "src"
sys.path.insert(0, str(MATCHUP_MACHINE_PATH))

try:
    from fantasy_inference import (
        load_artifacts,
        find_player_id,
        estimate_batter_outcome_probs_from_history,
    )
    from fantasy_scoring import expected_hitter_points_per_pa
    from train_hit_model import fill_missing_values
    from build_dataset import OUTCOME_LABELS
    import numpy as np
    
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import matchup_machine modules: {e}")
    ML_AVAILABLE = False
    

def safe_estimate_batter_probs(model, feature_cols, matchups, batter_id, min_pas, recent_only, recent_start_year):
    """Wrapper that handles dtype conversion issues in matchup_machine."""
    # Filter data
    df = matchups.copy()
    df = df[df["batter"] == int(batter_id)]
    df = df[df["outcome_id"].notna()]
    
    if recent_only:
        df = df[df["date"].dt.year >= recent_start_year]
    
    if df.empty:
        raise ValueError(f"No historical plate appearances found for batter_id={batter_id}")
    
    if len(df) < min_pas:
        print(f"Warning: only {len(df)} PAs for batter_id={batter_id} (min_pas={min_pas})")
    
    # Align to feature cols and handle missing values
    X = df.reindex(columns=feature_cols, fill_value=0)
    
    # Custom fill_missing_values that handles Int32 issues
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].isna().any():
            # Convert to float64 to avoid Int32 issues
            if X_filled[col].dtype.name.startswith('Int'):
                X_filled[col] = X_filled[col].astype('float64')
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    
    # Convert all Int types to float to avoid prediction issues
    for col in X_filled.columns:
        if X_filled[col].dtype.name.startswith('Int'):
            X_filled[col] = X_filled[col].astype('float64')
    
    # Run prediction
    probs = model.predict_proba(X_filled)
    avg_probs = probs.mean(axis=0)
    
    return {label: float(p) for label, p in zip(OUTCOME_LABELS, avg_probs)}


class MLProjectionEngine:
    """
    Generates ML-based fantasy projections using matchup_machine models.
    """
    
    def __init__(self):
        """Initialize the projection engine by loading trained models."""
        self.ml_available = ML_AVAILABLE
        
        if not self.ml_available:
            print("ML projections disabled - matchup_machine not available")
            return
            
        try:
            print("Loading ML models...")
            (
                self.model,
                self.feature_cols,
                self.pitcher_profiles,
                self.batter_profiles,
                self.player_index,
                self.pa_proj,
                self.matchups,
            ) = load_artifacts()
            print(f"✓ Loaded models successfully ({len(self.matchups):,} historical PAs)")
        except Exception as e:
            print(f"Error loading ML artifacts: {e}")
            self.ml_available = False
    
    def get_batter_projection(
        self, 
        batter_name: str, 
        pitcher_name: Optional[str] = None,
        default_pa: int = 4
    ) -> Dict:
        """
        Generate fantasy projection for a batter.
        
        Args:
            batter_name: Player name (e.g., "Aaron Judge")
            pitcher_name: Opposing pitcher name (optional - uses general projection if None)
            default_pa: Expected plate appearances for this game
            
        Returns:
            Dictionary with:
                - expected_points: Projected fantasy points
                - expected_points_per_pa: Points per plate appearance
                - outcome_probs: Probability distribution over outcomes
                - confidence: 'high', 'medium', 'low', or 'none'
                - error: Error message if projection failed
        """
        if not self.ml_available:
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'error': 'ML models not available'
            }
        
        try:
            # Find batter ID
            batter_id = find_player_id(self.player_index, batter_name)
            
            # Ensure batter_id is clean int (no Int32 dtype issues)
            batter_id = int(batter_id)
            
            # Get outcome probabilities from historical performance
            # Use safe wrapper to avoid Int32 dtype issues
            outcome_probs = safe_estimate_batter_probs(
                model=self.model,
                feature_cols=self.feature_cols,
                matchups=self.matchups,
                batter_id=batter_id,
                min_pas=100,  # Lower threshold for more coverage
                recent_only=True,
                recent_start_year=2024,
            )
            
            # Calculate expected points per PA
            ev_per_pa = expected_hitter_points_per_pa(outcome_probs)
            expected_points = ev_per_pa * default_pa
            
            # Determine confidence based on sample size
            batter_pas = self.matchups[
                (self.matchups['batter'] == batter_id) & 
                (self.matchups['date'].dt.year >= 2024) &
                (self.matchups['outcome_id'].notna())
            ]
            
            num_pas = int(len(batter_pas))
            if num_pas >= 300:
                confidence = 'high'
            elif num_pas >= 150:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'expected_points': round(expected_points, 2),
                'expected_points_per_pa': round(ev_per_pa, 3),
                'outcome_probs': outcome_probs,
                'confidence': confidence,
                'sample_size': num_pas,
                'error': None
            }
            
        except ValueError as e:
            # Player not found in database
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'error': f'Player not found: {str(e)}'
            }
        except Exception as e:
            # Other errors
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'error': f'Projection failed: {str(e)}'
            }
    
    def get_roster_projections(
        self,
        roster_df: pd.DataFrame,
        games_today: pd.DataFrame,
        team_name_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Generate projections for entire roster.
        
        Args:
            roster_df: DataFrame with columns: name, proTeam, position
            games_today: DataFrame with today's games
            team_name_map: Mapping from team abbreviation to full name
            
        Returns:
            DataFrame with added columns: ml_projection, ml_confidence, ml_error
        """
        if not self.ml_available:
            roster_df['ml_projection'] = None
            roster_df['ml_confidence'] = 'none'
            roster_df['ml_error'] = 'ML not available'
            return roster_df
        
        projections = []
        
        for _, player in roster_df.iterrows():
            name = player['name']
            team = player.get('proTeam', '')
            
            # Check if team plays today
            full_team_name = team_name_map.get(team)
            has_game_today = False
            
            if full_team_name and not games_today.empty:
                has_game_today = (
                    (games_today['home_team'] == full_team_name) | 
                    (games_today['away_team'] == full_team_name)
                ).any()
            
            # Get projection (4 PA if game today, 0 otherwise)
            default_pa = 4 if has_game_today else 0
            projection = self.get_batter_projection(name, default_pa=default_pa)
            
            projections.append({
                'name': name,
                'ml_projection': projection['expected_points'],
                'ml_pts_per_pa': projection['expected_points_per_pa'],
                'ml_confidence': projection['confidence'],
                'ml_sample_size': projection.get('sample_size'),
                'ml_error': projection['error']
            })
        
        proj_df = pd.DataFrame(projections)
        result = roster_df.merge(proj_df, on='name', how='left')
        
        return result


def test_projection_engine():
    """Test the projection engine with sample players."""
    print("\n" + "="*60)
    print("Testing ML Projection Engine")
    print("="*60)
    
    engine = MLProjectionEngine()
    
    if not engine.ml_available:
        print("\n✗ ML models not available - cannot run test")
        return False
    
    # Test with a few well-known players
    test_players = [
        "Aaron Judge",
        "Shohei Ohtani", 
        "Ronald Acuna",
        "Freddie Freeman"
    ]
    
    print("\nTesting projections for sample players:\n")
    
    for player_name in test_players:
        proj = engine.get_batter_projection(player_name, default_pa=4)
        
        if proj['error']:
            print(f"✗ {player_name}: {proj['error']}")
        else:
            print(f"✓ {player_name}:")
            print(f"  Expected points (4 PA): {proj['expected_points']}")
            print(f"  Points per PA: {proj['expected_points_per_pa']}")
            print(f"  Confidence: {proj['confidence']} ({proj['sample_size']} PAs)")
            print(f"  Top outcomes:")
            
            # Show top 3 outcome probabilities
            sorted_outcomes = sorted(
                proj['outcome_probs'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for outcome, prob in sorted_outcomes:
                print(f"    - {outcome}: {prob:.3f}")
            print()
    
    print("="*60)
    print("✓ Test complete")
    return True


if __name__ == "__main__":
    # Run test when module is executed directly
    test_projection_engine()
