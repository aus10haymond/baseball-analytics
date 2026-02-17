"""
Pitcher-Aware Projections - Phase 2

Generates matchup-specific fantasy projections based on today's probable pitchers.
Uses historical batter vs pitcher performance to adjust predictions.
"""

import sys
import requests
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import date
import pandas as pd
import numpy as np

# Add matchup_machine to path
MATCHUP_MACHINE_PATH = Path(__file__).parent.parent / "matchup_machine" / "src"
sys.path.insert(0, str(MATCHUP_MACHINE_PATH))

try:
    from fantasy_inference import load_artifacts, find_player_id
    from fantasy_scoring import expected_hitter_points_per_pa
    from build_dataset import OUTCOME_LABELS
    
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import matchup_machine modules: {e}")
    ML_AVAILABLE = False


def safe_predict_probs(model, X_filled):
    """Safe wrapper for model.predict_proba that handles dtype issues."""
    # Convert all Int types to float64
    for col in X_filled.columns:
        if X_filled[col].dtype.name.startswith('Int'):
            X_filled[col] = X_filled[col].astype('float64')
    
    probs = model.predict_proba(X_filled)
    return probs


class PitcherAwareEngine:
    """
    Advanced projection engine that uses today's probable pitchers
    for matchup-specific predictions.
    """
    
    def __init__(self):
        """Initialize the pitcher-aware engine by loading trained models."""
        self.ml_available = ML_AVAILABLE
        
        if not self.ml_available:
            print("ML projections disabled - matchup_machine not available")
            return
            
        try:
            print("Loading ML models for pitcher-aware projections...")
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
    
    def get_matchup_projection(
        self,
        batter_name: str,
        pitcher_name: str,
        default_pa: int = 4
    ) -> Dict:
        """
        Generate matchup-specific projection for batter vs specific pitcher.
        
        Args:
            batter_name: Batter's name
            pitcher_name: Opposing pitcher's name
            default_pa: Expected plate appearances
            
        Returns:
            Dictionary with projection, confidence, and matchup details
        """
        if not self.ml_available:
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'matchup_type': 'no_data',
                'error': 'ML models not available'
            }
        
        try:
            # Find player IDs
            batter_id = int(find_player_id(self.player_index, batter_name))
            pitcher_id = int(find_player_id(self.player_index, pitcher_name))
            
            # Get batter's historical PAs
            batter_pas = self.matchups[
                (self.matchups['batter'] == batter_id) &
                (self.matchups['date'].dt.year >= 2024) &
                (self.matchups['outcome_id'].notna())
            ].copy()
            
            if batter_pas.empty:
                return {
                    'expected_points': None,
                    'expected_points_per_pa': None,
                    'outcome_probs': None,
                    'confidence': 'none',
                    'matchup_type': 'no_batter_data',
                    'error': f'No historical data for {batter_name}'
                }
            
            # Check for head-to-head history
            h2h_pas = batter_pas[batter_pas['pitcher'] == pitcher_id]
            
            # Get pitcher profile
            pitcher_profile = self.pitcher_profiles[
                self.pitcher_profiles['pitcher'] == pitcher_id
            ]
            
            if not h2h_pas.empty:
                # Use head-to-head history if available
                matchup_type = 'head_to_head'
                sample_size = len(h2h_pas)
                projection_data = h2h_pas
            elif not pitcher_profile.empty:
                # Use pitcher profile with batter's general approach
                matchup_type = 'pitcher_profile'
                sample_size = len(batter_pas)
                
                # Merge pitcher profile into batter's PAs
                projection_data = batter_pas.merge(
                    pitcher_profile,
                    on='pitcher',
                    how='left',
                    suffixes=('', '_new')
                )
                
                # Update pitcher-specific columns with actual pitcher's profile
                pitcher_cols = [c for c in pitcher_profile.columns if c != 'pitcher']
                for col in pitcher_cols:
                    if col in projection_data.columns:
                        projection_data[col] = pitcher_profile[col].iloc[0]
            else:
                # Fallback to batter's general performance
                matchup_type = 'general'
                sample_size = len(batter_pas)
                projection_data = batter_pas
            
            # Generate prediction
            X = projection_data.reindex(columns=self.feature_cols, fill_value=0)
            
            # Handle missing values
            X_filled = X.copy()
            for col in X_filled.columns:
                if X_filled[col].isna().any():
                    if X_filled[col].dtype.name.startswith('Int'):
                        X_filled[col] = X_filled[col].astype('float64')
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
            
            # Run prediction
            probs = safe_predict_probs(self.model, X_filled)
            avg_probs = probs.mean(axis=0)
            
            outcome_probs = {label: float(p) for label, p in zip(OUTCOME_LABELS, avg_probs)}
            
            # Calculate expected points
            ev_per_pa = expected_hitter_points_per_pa(outcome_probs)
            expected_points = ev_per_pa * default_pa
            
            # Determine confidence
            if matchup_type == 'head_to_head' and sample_size >= 20:
                confidence = 'very_high'
            elif matchup_type == 'head_to_head' and sample_size >= 10:
                confidence = 'high'
            elif matchup_type == 'pitcher_profile' and sample_size >= 300:
                confidence = 'high'
            elif matchup_type == 'pitcher_profile' and sample_size >= 150:
                confidence = 'medium'
            elif sample_size >= 100:
                confidence = 'low'
            else:
                confidence = 'very_low'
            
            return {
                'expected_points': round(expected_points, 2),
                'expected_points_per_pa': round(ev_per_pa, 3),
                'outcome_probs': outcome_probs,
                'confidence': confidence,
                'matchup_type': matchup_type,
                'sample_size': sample_size,
                'h2h_sample': len(h2h_pas) if not h2h_pas.empty else 0,
                'pitcher_name': pitcher_name,
                'error': None
            }
            
        except ValueError as e:
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'matchup_type': 'error',
                'error': str(e)
            }
        except Exception as e:
            return {
                'expected_points': None,
                'expected_points_per_pa': None,
                'outcome_probs': None,
                'confidence': 'none',
                'matchup_type': 'error',
                'error': f'Projection failed: {str(e)}'
            }
    
    def get_todays_matchups(self) -> Dict[str, Dict]:
        """
        Fetch today's games and probable pitchers from MLB API.
        
        Returns:
            Dictionary mapping team name to opponent pitcher
        """
        today = date.today().strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"Error fetching today's matchups: {e}")
            return {}
        
        matchups = {}
        
        for date_info in data.get("dates", []):
            for game in date_info.get("games", []):
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]
                
                # Get probable pitchers
                home_pitcher = None
                away_pitcher = None
                
                if "probablePitcher" in game["teams"]["home"]:
                    home_pitcher = game["teams"]["home"]["probablePitcher"]["fullName"]
                
                if "probablePitcher" in game["teams"]["away"]:
                    away_pitcher = game["teams"]["away"]["probablePitcher"]["fullName"]
                
                # Map teams to opposing pitchers
                if away_pitcher:
                    matchups[home_team] = {
                        'opponent_pitcher': away_pitcher,
                        'opponent_team': away_team,
                        'is_home': True
                    }
                
                if home_pitcher:
                    matchups[away_team] = {
                        'opponent_pitcher': home_pitcher,
                        'opponent_team': home_team,
                        'is_home': False
                    }
        
        return matchups
    
    def get_roster_matchup_projections(
        self,
        roster_df: pd.DataFrame,
        team_name_map: Dict[str, str],
        default_pa: int = 4
    ) -> pd.DataFrame:
        """
        Generate pitcher-aware projections for entire roster.
        
        Args:
            roster_df: DataFrame with roster (name, proTeam columns)
            team_name_map: Mapping from abbreviation to full team name
            default_pa: Expected PAs per game
            
        Returns:
            DataFrame with pitcher-aware projections added
        """
        if not self.ml_available:
            roster_df['pa_projection'] = None
            roster_df['pa_confidence'] = 'none'
            roster_df['pa_matchup_type'] = 'no_data'
            roster_df['opponent_pitcher'] = None
            return roster_df
        
        # Get today's matchups
        todays_matchups = self.get_todays_matchups()
        
        projections = []
        
        for _, player in roster_df.iterrows():
            name = player['name']
            team = player.get('proTeam', '')
            
            # Get full team name
            full_team_name = team_name_map.get(team)
            
            # Check if team plays today and get opposing pitcher
            opponent_pitcher = None
            is_home = None
            matchup_info = None
            
            if full_team_name and full_team_name in todays_matchups:
                matchup_info = todays_matchups[full_team_name]
                opponent_pitcher = matchup_info['opponent_pitcher']
                is_home = matchup_info['is_home']
            
            # Generate projection
            if opponent_pitcher:
                proj = self.get_matchup_projection(name, opponent_pitcher, default_pa)
            else:
                # No game today or no probable pitcher
                proj = {
                    'expected_points': 0.0 if not matchup_info else None,
                    'expected_points_per_pa': None,
                    'confidence': 'none',
                    'matchup_type': 'no_game',
                    'h2h_sample': 0,
                    'sample_size': 0,
                    'error': 'No game today' if not matchup_info else 'No probable pitcher'
                }
            
            projections.append({
                'name': name,
                'pa_projection': proj['expected_points'],
                'pa_pts_per_pa': proj['expected_points_per_pa'],
                'pa_confidence': proj['confidence'],
                'pa_matchup_type': proj['matchup_type'],
                'pa_h2h_sample': proj.get('h2h_sample', 0),
                'pa_sample_size': proj.get('sample_size', 0),
                'opponent_pitcher': opponent_pitcher,
                'is_home': is_home,
                'pa_error': proj.get('error')
            })
        
        proj_df = pd.DataFrame(projections)
        result = roster_df.merge(proj_df, on='name', how='left')
        
        return result


def compare_projections(general_proj: float, matchup_proj: float) -> Dict:
    """
    Compare general projection vs matchup-specific projection.
    
    Returns analysis of matchup advantage/disadvantage.
    """
    if general_proj is None or matchup_proj is None:
        return {'advantage': 'unknown', 'difference': 0, 'pct_change': 0}
    
    diff = matchup_proj - general_proj
    pct_change = (diff / general_proj * 100) if general_proj > 0 else 0
    
    if pct_change > 15:
        advantage = 'strong_positive'
    elif pct_change > 5:
        advantage = 'positive'
    elif pct_change < -15:
        advantage = 'strong_negative'
    elif pct_change < -5:
        advantage = 'negative'
    else:
        advantage = 'neutral'
    
    return {
        'advantage': advantage,
        'difference': round(diff, 2),
        'pct_change': round(pct_change, 1)
    }


def test_pitcher_aware_engine():
    """Test the pitcher-aware projection engine."""
    print("\n" + "="*70)
    print("Testing Pitcher-Aware Projection Engine (Phase 2)")
    print("="*70)
    
    engine = PitcherAwareEngine()
    
    if not engine.ml_available:
        print("\n✗ ML models not available - cannot run test")
        return False
    
    # Test matchup-specific projections
    test_matchups = [
        ("Aaron Judge", "Gerrit Cole"),
        ("Shohei Ohtani", "Blake Snell"),
        ("Freddie Freeman", "Sandy Alcantara"),
    ]
    
    print("\nTesting matchup-specific projections:\n")
    
    for batter, pitcher in test_matchups:
        proj = engine.get_matchup_projection(batter, pitcher, default_pa=4)
        
        if proj['error']:
            print(f"✗ {batter} vs {pitcher}: {proj['error']}")
        else:
            print(f"✓ {batter} vs {pitcher}:")
            print(f"  Expected points (4 PA): {proj['expected_points']}")
            print(f"  Points per PA: {proj['expected_points_per_pa']}")
            print(f"  Matchup type: {proj['matchup_type']}")
            print(f"  Confidence: {proj['confidence']}")
            print(f"  H2H sample: {proj['h2h_sample']} PAs")
            print(f"  Total sample: {proj['sample_size']} PAs")
            
            # Show top 3 outcomes
            if proj['outcome_probs']:
                sorted_outcomes = sorted(
                    proj['outcome_probs'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                print(f"  Top outcomes:")
                for outcome, prob in sorted_outcomes:
                    print(f"    - {outcome}: {prob:.3f}")
            print()
    
    print("="*70)
    print("✓ Pitcher-aware test complete")
    return True


if __name__ == "__main__":
    test_pitcher_aware_engine()
