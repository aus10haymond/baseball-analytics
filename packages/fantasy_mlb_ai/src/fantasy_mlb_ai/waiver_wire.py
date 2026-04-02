"""
Waiver Wire Recommendations

Identifies undervalued free agents and suggests add/drop pairs
based on rest-of-season projections.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import date, timedelta
import requests
from pathlib import Path

try:
    from .pitcher_aware_projections import PitcherAwareEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class WaiverWireAnalyzer:
    """
    Analyzes waiver wire to find undervalued free agents.
    """

    def __init__(self, espn_league_id: Optional[str] = None,
                 espn_swid: Optional[str] = None,
                 espn_s2: Optional[str] = None):
        """
        Initialize waiver wire analyzer.

        Args:
            espn_league_id: ESPN league ID
            espn_swid: ESPN SWID cookie
            espn_s2: ESPN s2 cookie
        """
        self.league_id = espn_league_id
        self.swid = espn_swid
        self.s2 = espn_s2
        self.engine = PitcherAwareEngine() if ENGINE_AVAILABLE else None

    def fetch_available_players(
        self,
        position: Optional[str] = None,
        max_ownership: float = 50.0,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch available free agents from ESPN.

        Args:
            position: Filter by position (e.g., 'OF', '1B')
            max_ownership: Maximum ownership percentage
            limit: Maximum number of players to fetch

        Returns:
            DataFrame with available players
        """
        if not self.league_id:
            print("ESPN league ID not configured")
            return pd.DataFrame()

        # ESPN API endpoint for available players
        url = f"https://fantasy.espn.com/apis/v3/games/flb/seasons/2026/segments/0/leagues/{self.league_id}"

        params = {
            'view': 'kona_player_info'
        }

        cookies = {}
        if self.swid:
            cookies['SWID'] = self.swid
        if self.s2:
            cookies['espn_s2'] = self.s2

        try:
            response = requests.get(url, params=params, cookies=cookies, timeout=15)
            data = response.json()

            players = []
            for player_data in data.get('players', []):
                player = player_data.get('player', {})

                # Skip if not available
                if player.get('onTeamId'):
                    continue

                # Get player info
                full_name = player.get('fullName', '')
                ownership = player.get('ownership', {}).get('percentOwned', 0)

                # Filter by ownership
                if ownership > max_ownership:
                    continue

                # Get position
                eligible_slots = player.get('eligibleSlots', [])
                pos = eligible_slots[0] if eligible_slots else 'Unknown'

                # Filter by position if specified
                if position and position not in eligible_slots:
                    continue

                # Get stats
                stats = player.get('stats', [])
                recent_stats = {}
                if stats:
                    # Get most recent stats
                    recent_stats = stats[-1].get('stats', {})

                players.append({
                    'player_id': player.get('id'),
                    'name': full_name,
                    'position': pos,
                    'ownership_pct': ownership,
                    'team': player.get('proTeam', ''),
                    'recent_stats': recent_stats
                })

                if len(players) >= limit:
                    break

            return pd.DataFrame(players)

        except Exception as e:
            print(f"Error fetching available players: {e}")
            return pd.DataFrame()

    def project_ros_value(
        self,
        players_df: pd.DataFrame,
        weeks_remaining: int = 12
    ) -> pd.DataFrame:
        """
        Project rest-of-season value for players.

        Args:
            players_df: DataFrame with player information
            weeks_remaining: Number of weeks remaining in season

        Returns:
            DataFrame with ROS projections added
        """
        if self.engine is None or not self.engine.ml_available:
            print("ML engine not available for projections")
            players_df['ros_projection'] = None
            return players_df

        projections = []

        for _, player in players_df.iterrows():
            name = player['name']

            # Get average projection (simplified - could factor in schedule strength)
            try:
                # Use general projection (no specific pitcher)
                proj = self.engine.get_matchup_projection(
                    batter_name=name,
                    pitcher_name="",  # Will use general performance
                    default_pa=4
                )

                if proj['error'] is None and proj['expected_points_per_pa']:
                    # Project for rest of season
                    # Assume 6 games per week, 4 PA per game
                    games_remaining = weeks_remaining * 6
                    pa_remaining = games_remaining * 4

                    ros_proj = proj['expected_points_per_pa'] * pa_remaining

                    projections.append({
                        'name': name,
                        'ros_projection': ros_proj,
                        'pts_per_game': proj['expected_points_per_pa'] * 4,
                        'confidence': proj['confidence']
                    })
                else:
                    projections.append({
                        'name': name,
                        'ros_projection': None,
                        'pts_per_game': None,
                        'confidence': 'none'
                    })

            except Exception as e:
                print(f"Error projecting {name}: {e}")
                projections.append({
                    'name': name,
                    'ros_projection': None,
                    'pts_per_game': None,
                    'confidence': 'none'
                })

        proj_df = pd.DataFrame(projections)
        result = players_df.merge(proj_df, on='name', how='left')

        return result

    def suggest_adds(
        self,
        available_df: pd.DataFrame,
        top_n: int = 20,
        min_ownership: float = 5.0
    ) -> pd.DataFrame:
        """
        Suggest top waiver wire adds.

        Args:
            available_df: DataFrame with available players and projections
            top_n: Number of suggestions to return
            min_ownership: Minimum ownership to consider (avoid untested players)

        Returns:
            DataFrame with top suggestions
        """
        # Filter
        filtered = available_df[
            (available_df['ros_projection'].notna()) &
            (available_df['ownership_pct'] >= min_ownership)
        ].copy()

        # Calculate value score (projection relative to ownership)
        filtered['value_score'] = (
            filtered['ros_projection'] / (filtered['ownership_pct'] + 1)
        )

        # Sort by value score
        suggestions = filtered.nlargest(top_n, 'value_score')

        return suggestions[[
            'name', 'position', 'ownership_pct', 'ros_projection',
            'pts_per_game', 'confidence', 'value_score'
        ]]

    def suggest_drops(
        self,
        roster_df: pd.DataFrame,
        bottom_n: int = 10
    ) -> pd.DataFrame:
        """
        Suggest weakest players on roster to drop.

        Args:
            roster_df: DataFrame with current roster and projections
            bottom_n: Number of suggestions to return

        Returns:
            DataFrame with drop candidates
        """
        # Filter out IL players and those with no projections
        droppable = roster_df[
            (roster_df['lineupSlot'] != 'IL') &
            (roster_df.get('ml_projection', pd.Series()).notna())
        ].copy()

        if droppable.empty:
            return pd.DataFrame()

        # Sort by projection (ascending)
        suggestions = droppable.nsmallest(bottom_n, 'ml_projection')

        return suggestions[[
            'name', 'position', 'proTeam', 'ml_projection',
            'ml_confidence', 'lineupSlot'
        ]]

    def generate_add_drop_pairs(
        self,
        roster_df: pd.DataFrame,
        available_df: pd.DataFrame,
        min_improvement: float = 10.0
    ) -> List[Dict]:
        """
        Generate add/drop pair recommendations.

        Args:
            roster_df: Current roster with projections
            available_df: Available players with projections
            min_improvement: Minimum ROS improvement to suggest (in points)

        Returns:
            List of add/drop pair recommendations
        """
        drops = self.suggest_drops(roster_df, bottom_n=15)
        adds = self.suggest_adds(available_df, top_n=30)

        if drops.empty or adds.empty:
            return []

        pairs = []

        for _, drop_player in drops.iterrows():
            drop_pos = drop_player['position']
            drop_proj = drop_player['ml_projection']

            # Find adds at same position
            same_pos_adds = adds[adds['position'] == drop_pos]

            for _, add_player in same_pos_adds.iterrows():
                add_proj = add_player['ros_projection']

                # Annualize drop projection for comparison
                # (current projection is per game, ROS is total)
                weeks_remaining = 12
                drop_ros = drop_proj * 6 * weeks_remaining  # 6 games per week

                improvement = add_proj - drop_ros

                if improvement >= min_improvement:
                    pairs.append({
                        'drop_name': drop_player['name'],
                        'drop_projection': drop_ros,
                        'add_name': add_player['name'],
                        'add_projection': add_proj,
                        'add_ownership': add_player['ownership_pct'],
                        'improvement': improvement,
                        'position': drop_pos,
                        'confidence': add_player['confidence']
                    })

        # Sort by improvement
        pairs = sorted(pairs, key=lambda x: x['improvement'], reverse=True)

        return pairs[:10]  # Top 10 pairs


def main():
    """
    Demo waiver wire analysis.
    """
    print("=" * 70)
    print("Waiver Wire Analysis")
    print("=" * 70)

    # Load roster
    data_dir = Path(__file__).parent.parent.parent / "data"
    roster_path = data_dir / "my_roster.csv"

    if not roster_path.exists():
        print("\nNo roster file found. Run fetch_daily_data.py first.")
        return

    roster_df = pd.read_csv(roster_path)
    print(f"\nLoaded roster: {len(roster_df)} players")

    # Initialize analyzer
    analyzer = WaiverWireAnalyzer()

    # For demo, create sample available players
    # In production, would fetch from ESPN API
    print("\nNote: ESPN API integration requires league credentials.")
    print("See .env.example for configuration.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
