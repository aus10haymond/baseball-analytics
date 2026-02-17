# src/fantasy_cli.py

from __future__ import annotations

import argparse

from .fantasy_scoring import expected_hitter_points_per_pa
from .fantasy_inference import (
    load_artifacts,
    find_player_id,
    lookup_projected_pa,
    estimate_batter_outcome_probs_from_history,
)


def cmd_hitter(args) -> None:
    """
    CLI command: evaluate a hitter's fantasy value for 2026
    based on:
      - projected PA for 2026 (from historical data)
      - per-PA outcome probabilities averaged over their historical PAs
      - scoring rules in fantasy_scoring.py
    """
    (
        model,
        feature_cols,
        pitcher_profiles,   # unused in this command for now, but available
        batter_profiles,    # unused for now
        player_index,
        pa_proj,
        matchups,
    ) = load_artifacts()

    # 1) Resolve name -> batter_id
    batter_id = find_player_id(player_index, args.player)

    # 2) Estimate outcome probabilities from historical PAs
    outcome_probs = estimate_batter_outcome_probs_from_history(
        model=model,
        feature_cols=feature_cols,
        matchups=matchups,
        batter_id=batter_id,
        min_pas=200,        # tweak if you want more/less strict
        recent_only=True,
        recent_start_year=2024,
    )

    # 3) Compute EV per PA
    ev_per_pa = expected_hitter_points_per_pa(outcome_probs)

    # 4) Look up projected PA for 2026 and season total
    projected_pa = lookup_projected_pa(batter_id, pa_proj)
    season_points = ev_per_pa * projected_pa

    print(f"\nPlayer: {args.player}")
    print(f"Projected PA (2026):       {projected_pa}")
    print(f"Expected points per PA:    {ev_per_pa:.3f}")
    print(f"Projected season points:   {season_points:.1f}\n")

    print("Outcome probabilities (per PA):")
    for label, prob in outcome_probs.items():
        print(f"  {label:17s}: {prob:6.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fantasy MLB helper using matchup_machine outcome models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Hitter evaluation command
    p_hitter = subparsers.add_parser(
        "hitter", help="Evaluate a hitter's fantasy value for the 2026 season."
    )
    p_hitter.add_argument(
        "player",
        type=str,
        help="Player name substring (e.g., 'Carroll', 'Ohtani').",
    )
    p_hitter.set_defaults(func=cmd_hitter)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
