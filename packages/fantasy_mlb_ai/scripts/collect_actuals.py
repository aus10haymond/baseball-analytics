"""
Collect Actual Results

Fetches actual game stats from MLB Stats API and updates prediction tracker
with actual fantasy points earned. Run this daily after games complete.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import time

# Add parent to path for imports
parent_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(parent_dir))

from fantasy_mlb_ai.prediction_tracker import PredictionTracker


# Fantasy scoring (from matchup_machine/fantasy_scoring.py)
FANTASY_SCORING = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'home_run': 6,
    'walk': 1,
    'strikeout': -1,
    'rbi': 1,
    'run': 1,
    'stolen_base': 2,
    'caught_stealing': -1
}


def get_player_stats_for_date(player_name: str, game_date: str) -> Optional[Dict]:
    """
    Fetch a player's stats for a specific date from MLB Stats API.

    Args:
        player_name: Player's full name
        game_date: Date in YYYY-MM-DD format

    Returns:
        Dictionary with stats or None if not found
    """
    # Search for player
    search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={player_name}"

    try:
        response = requests.get(search_url, timeout=10)
        data = response.json()

        if not data.get('people'):
            print(f"  Player not found: {player_name}")
            return None

        player_id = data['people'][0]['id']

        # Get game stats for the date
        stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={game_date[:4]}&group=hitting"

        response = requests.get(stats_url, timeout=10)
        stats_data = response.json()

        # Find stats for the specific date
        for split in stats_data.get('stats', []):
            for game_stat in split.get('splits', []):
                if game_stat.get('date') == game_date:
                    return game_stat['stat']

        # No game on this date
        return None

    except Exception as e:
        print(f"  Error fetching stats for {player_name}: {e}")
        return None


def calculate_fantasy_points(stats: Dict) -> float:
    """
    Calculate fantasy points from MLB stats.

    Args:
        stats: Dictionary with hitting stats

    Returns:
        Total fantasy points
    """
    points = 0.0

    # Plate appearance outcomes
    points += stats.get('hits', 0) * 1  # singles
    points += stats.get('doubles', 0) * 2
    points += stats.get('triples', 0) * 3
    points += stats.get('homeRuns', 0) * 6
    points += stats.get('baseOnBalls', 0) * 1
    points += stats.get('strikeOuts', 0) * -1

    # Additional stats
    points += stats.get('rbi', 0) * 1
    points += stats.get('runs', 0) * 1
    points += stats.get('stolenBases', 0) * 2
    points += stats.get('caughtStealing', 0) * -1

    return points


def collect_actuals_for_date(
    tracker: PredictionTracker,
    target_date: str,
    verbose: bool = True
) -> int:
    """
    Collect actual results for all predictions on a given date.

    Args:
        tracker: PredictionTracker instance
        target_date: Date in YYYY-MM-DD format
        verbose: Print progress

    Returns:
        Number of actuals collected
    """
    if verbose:
        print(f"\nCollecting actuals for {target_date}...")

    # Get predictions without actuals
    preds = tracker.get_predictions(
        start_date=target_date,
        end_date=target_date,
        only_with_actuals=False
    )

    # Filter to only those without actuals
    preds_to_update = preds[preds['actual_points'].isna()]

    if preds_to_update.empty:
        if verbose:
            print("  All predictions already have actuals.")
        return 0

    if verbose:
        print(f"  Found {len(preds_to_update)} predictions to update")

    updated = 0

    for _, pred in preds_to_update.iterrows():
        player_name = pred['player_name']

        if verbose:
            print(f"  Fetching stats for {player_name}...", end=" ")

        stats = get_player_stats_for_date(player_name, target_date)

        if stats:
            # Calculate fantasy points
            actual_points = calculate_fantasy_points(stats)
            pa_actual = stats.get('plateAppearances', 0)

            # Update tracker
            tracker.update_actual(
                date_str=target_date,
                player_name=player_name,
                actual_points=actual_points,
                pa_actual=pa_actual
            )

            if verbose:
                print(f"✓ {actual_points:.1f} pts ({pa_actual} PA)")

            updated += 1
        else:
            if verbose:
                print("No game data")

        # Rate limiting
        time.sleep(0.2)

    if verbose:
        print(f"\n  Updated {updated} predictions")

    return updated


def main():
    """
    Main entry point - collect actuals for yesterday.
    """
    print("=" * 70)
    print("Collecting Actual Results")
    print("=" * 70)

    # Default to yesterday
    target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Allow override via command line
    if len(sys.argv) > 1:
        target_date = sys.argv[1]

    print(f"\nTarget date: {target_date}")

    tracker = PredictionTracker()

    try:
        updated = collect_actuals_for_date(tracker, target_date, verbose=True)

        if updated > 0:
            # Calculate and save daily summary
            print(f"\nCalculating accuracy summary...")
            tracker.save_daily_summary(target_date)
            print("✓ Daily summary saved")

            # Show quick accuracy stats
            accuracy = tracker.calculate_accuracy(days=1)
            if accuracy:
                print(f"\nAccuracy for {target_date}:")
                print(f"  MAE: {accuracy['mae']:.2f}")
                print(f"  RMSE: {accuracy['rmse']:.2f}")
                print(f"  Sample size: {accuracy['sample_size']}")

    finally:
        tracker.close()

    print("\n" + "=" * 70)
    print(f"✓ Actuals collection complete ({updated} updated)")
    print("=" * 70)


if __name__ == "__main__":
    main()
