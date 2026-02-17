"""
Draft Rankings Generator for 2026 Fantasy Baseball

Uses matchup_machine ML models and 2026 MLB schedule to generate
season-long projections and draft rankings for batters and pitchers.

Scoring System:
  Batters: TB=1, RBI=1, R=1, SB=1, BB=1, K=-1
  Pitchers: IP=3, W=2, L=-2, HD=2, SV=5, ER=-2, H=-1, K=1, BB=-1
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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


# Team name normalization (schedule uses various formats)
TEAM_NAME_MAP = {
    'D-backs': 'Arizona Diamondbacks',
    'Diamondbacks': 'Arizona Diamondbacks',
    'Athletics': 'Oakland Athletics',
    "A's": 'Oakland Athletics',
    'Braves': 'Atlanta Braves',
    'Orioles': 'Baltimore Orioles',
    'Red Sox': 'Boston Red Sox',
    'Cubs': 'Chicago Cubs',
    'White Sox': 'Chicago White Sox',
    'Reds': 'Cincinnati Reds',
    'Guardians': 'Cleveland Guardians',
    'Rockies': 'Colorado Rockies',
    'Tigers': 'Detroit Tigers',
    'Astros': 'Houston Astros',
    'Royals': 'Kansas City Royals',
    'Angels': 'Los Angeles Angels',
    'Dodgers': 'Los Angeles Dodgers',
    'Marlins': 'Miami Marlins',
    'Brewers': 'Milwaukee Brewers',
    'Twins': 'Minnesota Twins',
    'Mets': 'New York Mets',
    'Yankees': 'New York Yankees',
    'Phillies': 'Philadelphia Phillies',
    'Pirates': 'Pittsburgh Pirates',
    'Padres': 'San Diego Padres',
    'Giants': 'San Francisco Giants',
    'Mariners': 'Seattle Mariners',
    'Cardinals': 'St. Louis Cardinals',
    'Rays': 'Tampa Bay Rays',
    'Rangers': 'Texas Rangers',
    'Blue Jays': 'Toronto Blue Jays',
    'Nationals': 'Washington Nationals',
}


class DraftRankingEngine:
    """
    Generates 2026 draft rankings using ML projections and full season schedule.
    """
    
    def __init__(self, schedule_path: str = "../matchup_machine/data/mlb_2026.csv"):
        """Initialize engine and load ML models and schedule."""
        self.ml_available = ML_AVAILABLE
        
        if not self.ml_available:
            print("❌ ML models not available - cannot generate rankings")
            return
        
        print("Loading ML models and data...")
        try:
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
            print(f"❌ Error loading ML artifacts: {e}")
            self.ml_available = False
            return
        
        # Load 2026 schedule
        print(f"Loading 2026 schedule from {schedule_path}...")
        self.schedule = pd.read_csv(schedule_path)
        self.schedule['Game Date'] = pd.to_datetime(self.schedule['Game Date'])
        print(f"✓ Loaded {len(self.schedule):,} games for 2026 season")
        
        # Build team schedule lookup
        self._build_team_schedules()
        
        # Get active players
        self._identify_active_players()
    
    def _build_team_schedules(self):
        """Build dictionary of each team's opponents throughout season."""
        self.team_schedules = defaultdict(list)
        
        for _, game in self.schedule.iterrows():
            home_team = TEAM_NAME_MAP.get(game['Home Team'], game['Home Team'])
            away_team = TEAM_NAME_MAP.get(game['Away Team'], game['Away Team'])
            game_date = game['Game Date']
            
            # Home team faces away team
            self.team_schedules[home_team].append({
                'date': game_date,
                'opponent': away_team,
                'home': True
            })
            
            # Away team faces home team
            self.team_schedules[away_team].append({
                'date': game_date,
                'opponent': home_team,
                'home': False
            })
    
    def _identify_active_players(self):
        """Identify players with significant 2024-2025 data."""
        # Filter to players with recent activity
        recent_pas = self.matchups[
            (self.matchups['date'].dt.year >= 2024) &
            (self.matchups['outcome_id'].notna())
        ]
        
        # Count PAs per batter
        batter_counts = recent_pas['batter'].value_counts()
        self.active_batters = set(batter_counts[batter_counts >= 100].index)
        
        # Count PAs per pitcher
        pitcher_counts = recent_pas['pitcher'].value_counts()
        self.active_pitchers = set(pitcher_counts[pitcher_counts >= 100].index)
        
        print(f"✓ Identified {len(self.active_batters)} active batters")
        print(f"✓ Identified {len(self.active_pitchers)} active pitchers")
    
    def _safe_predict_probs(self, batter_id: int) -> Optional[Dict[str, float]]:
        """Get outcome probabilities for a batter with dtype safety."""
        try:
            df = self.matchups[
                (self.matchups['batter'] == batter_id) &
                (self.matchups['date'].dt.year >= 2024) &
                (self.matchups['outcome_id'].notna())
            ].copy()
            
            if df.empty or len(df) < 50:
                return None
            
            X = df.reindex(columns=self.feature_cols, fill_value=0)
            
            # Handle Int32 dtype issues
            X_filled = X.copy()
            for col in X_filled.columns:
                if X_filled[col].isna().any():
                    if X_filled[col].dtype.name.startswith('Int'):
                        X_filled[col] = X_filled[col].astype('float64')
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
            
            for col in X_filled.columns:
                if X_filled[col].dtype.name.startswith('Int'):
                    X_filled[col] = X_filled[col].astype('float64')
            
            probs = self.model.predict_proba(X_filled)
            avg_probs = probs.mean(axis=0)
            
            return {label: float(p) for label, p in zip(OUTCOME_LABELS, avg_probs)}
        except Exception as e:
            return None
    
    def project_batter_season(
        self,
        batter_id: int,
        team_name: str,
        expected_pa_per_game: float = 4.0
    ) -> Optional[Dict]:
        """
        Project full season stats for a batter based on team schedule.
        
        Args:
            batter_id: Player ID
            team_name: Team name (normalized)
            expected_pa_per_game: Expected PAs per game
            
        Returns:
            Dict with season projections or None if failed
        """
        # Get outcome probabilities
        outcome_probs = self._safe_predict_probs(batter_id)
        if not outcome_probs:
            return None
        
        # Get points per PA
        pts_per_pa = expected_hitter_points_per_pa(outcome_probs)
        
        # Get team's schedule
        if team_name not in self.team_schedules:
            return None
        
        games = self.team_schedules[team_name]
        total_games = len(games)
        
        # Project season totals
        total_pa = total_games * expected_pa_per_game
        season_points = pts_per_pa * total_pa
        
        # Get player name
        player_row = self.player_index[self.player_index['player_id'] == batter_id]
        player_name = player_row['player_name'].iloc[0] if not player_row.empty else f"ID_{batter_id}"
        
        # Calculate component stats from outcome probabilities
        # Approximate based on typical outcome distributions
        singles = outcome_probs.get('single', 0) * total_pa
        doubles = outcome_probs.get('double', 0) * total_pa
        triples = outcome_probs.get('triple', 0) * total_pa
        hrs = outcome_probs.get('home_run', 0) * total_pa
        walks = outcome_probs.get('walk', 0) * total_pa
        strikeouts = outcome_probs.get('strikeout', 0) * total_pa
        
        # Estimate runs and RBIs (rough approximation)
        # HR ~ 1 run + 1 RBI, other hits ~0.3 runs, ~0.35 RBI
        est_runs = hrs + (singles + doubles * 1.5 + triples * 2) * 0.3
        est_rbis = hrs + (singles + doubles + triples) * 0.35
        
        return {
            'player_id': batter_id,
            'player_name': player_name.title(),
            'team': team_name,
            'games': total_games,
            'projected_pa': int(total_pa),
            'projected_points': round(season_points, 1),
            'pts_per_pa': round(pts_per_pa, 3),
            'singles': round(singles, 1),
            'doubles': round(doubles, 1),
            'triples': round(triples, 1),
            'home_runs': round(hrs, 1),
            'walks': round(walks, 1),
            'strikeouts': round(strikeouts, 1),
            'est_runs': round(est_runs, 1),
            'est_rbis': round(est_rbis, 1),
            'outcome_probs': outcome_probs
        }
    
    def project_pitcher_season(
        self,
        pitcher_id: int,
        team_name: str,
        is_starter: bool = True
    ) -> Optional[Dict]:
        """
        Project full season stats for a pitcher (inverse batter approach).
        
        Args:
            pitcher_id: Pitcher ID
            team_name: Team name
            is_starter: True for SP, False for RP
            
        Returns:
            Dict with season projections or None if failed
        """
        try:
            # Get all PAs where this pitcher faced batters
            pitcher_pas = self.matchups[
                (self.matchups['pitcher'] == pitcher_id) &
                (self.matchups['date'].dt.year >= 2024) &
                (self.matchups['outcome_id'].notna())
            ].copy()
            
            if pitcher_pas.empty or len(pitcher_pas) < 100:
                return None
            
            # Get actual outcome distribution
            outcome_counts = pitcher_pas['outcome_id'].value_counts()
            total_pas = len(pitcher_pas)
            
            outcome_probs = {}
            for i, label in enumerate(OUTCOME_LABELS):
                outcome_probs[label] = outcome_counts.get(i, 0) / total_pas
            
            # Project season based on role
            games = self.team_schedules.get(team_name, [])
            total_games = len(games)
            
            if is_starter:
                # Starters: ~32 starts, ~100 pitches = ~25 batters per start
                starts = min(32, total_games // 5)
                batters_faced = starts * 25
                innings = starts * 5.5  # Average ~5.5 IP per start
            else:
                # Relievers: ~65 appearances, ~15 pitches = ~5 batters per appearance
                appearances = min(65, int(total_games * 0.4))
                batters_faced = appearances * 5
                innings = appearances * 1.0  # Average 1 IP per appearance
            
            # Calculate outcomes
            hits = (outcome_probs.get('single', 0) + 
                   outcome_probs.get('double', 0) + 
                   outcome_probs.get('triple', 0) + 
                   outcome_probs.get('home_run', 0)) * batters_faced
            
            walks_issued = outcome_probs.get('walk', 0) * batters_faced
            strikeouts = outcome_probs.get('strikeout', 0) * batters_faced
            hrs_allowed = outcome_probs.get('home_run', 0) * batters_faced
            
            # Estimate ER (rough: ~0.13 ER per hit, 0.05 per walk, 0.35 per HR)
            est_er = hits * 0.13 + walks_issued * 0.05 + hrs_allowed * 0.35
            
            # Estimate wins/losses (very rough - depends on team quality)
            # Assume league average: 50% win rate for starters who go 5+ IP
            if is_starter:
                est_wins = starts * 0.50 * 0.6  # 60% of starts qualify for decision
                est_losses = starts * 0.50 * 0.4
                est_holds = 0
                est_saves = 0
            else:
                est_wins = appearances * 0.08  # Reliever win rate
                est_losses = appearances * 0.06
                est_holds = appearances * 0.15  # Hold opportunity rate
                est_saves = appearances * 0.10 if not is_starter else 0  # Closer rate
            
            # Calculate fantasy points
            points = (
                innings * 3 +
                est_wins * 2 +
                est_losses * -2 +
                est_holds * 2 +
                est_saves * 5 +
                est_er * -2 +
                hits * -1 +
                strikeouts * 1 +
                walks_issued * -1
            )
            
            # Get player name
            player_row = self.player_index[self.player_index['player_id'] == pitcher_id]
            player_name = player_row['player_name'].iloc[0] if not player_row.empty else f"ID_{pitcher_id}"
            
            return {
                'player_id': pitcher_id,
                'player_name': player_name.title(),
                'team': team_name,
                'role': 'SP' if is_starter else 'RP',
                'games': starts if is_starter else appearances,
                'innings': round(innings, 1),
                'projected_points': round(points, 1),
                'strikeouts': round(strikeouts, 1),
                'walks': round(walks_issued, 1),
                'hits_allowed': round(hits, 1),
                'earned_runs': round(est_er, 1),
                'est_wins': round(est_wins, 1),
                'est_losses': round(est_losses, 1),
                'est_holds': round(est_holds, 1),
                'est_saves': round(est_saves, 1),
                'era': round((est_er / innings * 9) if innings > 0 else 0, 2),
                'whip': round((hits + walks_issued) / innings if innings > 0 else 0, 2)
            }
        except Exception as e:
            return None
    
    def generate_batter_rankings(self, top_n: int = 300) -> pd.DataFrame:
        """Generate top N batter rankings."""
        print(f"\nGenerating batter rankings...")
        
        projections = []
        processed = 0
        
        for batter_id in self.active_batters:
            # Get player's most recent team (approximate from matchups)
            recent_games = self.matchups[
                (self.matchups['batter'] == batter_id) &
                (self.matchups['date'].dt.year >= 2024)
            ]
            
            if recent_games.empty:
                continue
            
            # Try common teams (this is a simplification - ideally get from rosters)
            team = "Unknown"
            for team_name in TEAM_NAME_MAP.values():
                projection = self.project_batter_season(batter_id, team_name)
                if projection and projection['projected_points'] > 0:
                    projections.append(projection)
                    processed += 1
                    break
            
            if processed % 50 == 0:
                print(f"  Processed {processed} batters...")
            
            if len(projections) >= top_n * 2:  # Get more than needed, then filter
                break
        
        df = pd.DataFrame(projections)
        if df.empty:
            print("❌ No batter projections generated")
            return df
        
        # Sort and rank
        df = df.sort_values('projected_points', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df.head(top_n)
    
    def generate_pitcher_rankings(self, top_n_sp: int = 100, top_n_rp: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate top N pitcher rankings (separate for SP and RP)."""
        print(f"\nGenerating pitcher rankings...")
        
        sp_projections = []
        rp_projections = []
        processed = 0
        
        for pitcher_id in self.active_pitchers:
            # Try as starter
            for team_name in TEAM_NAME_MAP.values():
                projection = self.project_pitcher_season(pitcher_id, team_name, is_starter=True)
                if projection and projection['projected_points'] > 0:
                    sp_projections.append(projection)
                    processed += 1
                    break
            
            # Also try as reliever
            for team_name in TEAM_NAME_MAP.values():
                projection = self.project_pitcher_season(pitcher_id, team_name, is_starter=False)
                if projection and projection['projected_points'] > 0:
                    rp_projections.append(projection)
                    break
            
            if processed % 50 == 0:
                print(f"  Processed {processed} pitchers...")
            
            if len(sp_projections) >= top_n_sp * 2 and len(rp_projections) >= top_n_rp * 2:
                break
        
        # Create DataFrames
        sp_df = pd.DataFrame(sp_projections)
        rp_df = pd.DataFrame(rp_projections)
        
        if not sp_df.empty:
            sp_df = sp_df.sort_values('projected_points', ascending=False).reset_index(drop=True)
            sp_df['rank'] = range(1, len(sp_df) + 1)
            sp_df = sp_df.head(top_n_sp)
        
        if not rp_df.empty:
            rp_df = rp_df.sort_values('projected_points', ascending=False).reset_index(drop=True)
            rp_df['rank'] = range(1, len(rp_df) + 1)
            rp_df = rp_df.head(top_n_rp)
        
        return sp_df, rp_df


def main():
    """Generate and save 2026 draft rankings."""
    print("="*70)
    print("2026 FANTASY BASEBALL DRAFT RANKINGS GENERATOR")
    print("="*70)
    
    engine = DraftRankingEngine()
    
    if not engine.ml_available:
        print("\n❌ Cannot generate rankings without ML models")
        return
    
    # Generate rankings
    batters_df = engine.generate_batter_rankings(top_n=300)
    sp_df, rp_df = engine.generate_pitcher_rankings(top_n_sp=100, top_n_rp=50)
    
    # Save results
    output_dir = Path("data/draft_rankings")
    output_dir.mkdir(exist_ok=True)
    
    if not batters_df.empty:
        batters_df.to_csv(output_dir / "batters_2026.csv", index=False)
        print(f"\n✓ Saved top {len(batters_df)} batters to {output_dir}/batters_2026.csv")
        
        print("\n" + "="*70)
        print("TOP 20 BATTERS - 2026 PROJECTIONS")
        print("="*70)
        display_cols = ['rank', 'player_name', 'team', 'projected_points', 
                       'home_runs', 'est_runs', 'est_rbis', 'walks', 'strikeouts']
        print(batters_df[display_cols].head(20).to_string(index=False))
    
    if not sp_df.empty:
        sp_df.to_csv(output_dir / "starters_2026.csv", index=False)
        print(f"\n✓ Saved top {len(sp_df)} starting pitchers to {output_dir}/starters_2026.csv")
        
        print("\n" + "="*70)
        print("TOP 20 STARTING PITCHERS - 2026 PROJECTIONS")
        print("="*70)
        display_cols = ['rank', 'player_name', 'team', 'projected_points',
                       'innings', 'strikeouts', 'era', 'whip', 'est_wins']
        print(sp_df[display_cols].head(20).to_string(index=False))
    
    if not rp_df.empty:
        rp_df.to_csv(output_dir / "relievers_2026.csv", index=False)
        print(f"\n✓ Saved top {len(rp_df)} relief pitchers to {output_dir}/relievers_2026.csv")
        
        print("\n" + "="*70)
        print("TOP 20 RELIEF PITCHERS - 2026 PROJECTIONS")
        print("="*70)
        display_cols = ['rank', 'player_name', 'team', 'projected_points',
                       'games', 'est_saves', 'est_holds', 'strikeouts', 'era']
        print(rp_df[display_cols].head(20).to_string(index=False))
    
    print("\n" + "="*70)
    print("DRAFT RANKINGS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
