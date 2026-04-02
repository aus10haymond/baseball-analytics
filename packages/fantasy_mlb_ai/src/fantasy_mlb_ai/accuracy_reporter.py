"""
Accuracy Reporting System

Generates reports on prediction accuracy with breakdowns by player,
confidence level, and matchup type.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from .prediction_tracker import PredictionTracker


class AccuracyReporter:
    """
    Generates accuracy reports from prediction tracking data.
    """

    def __init__(self, tracker: Optional[PredictionTracker] = None):
        """
        Initialize accuracy reporter.

        Args:
            tracker: PredictionTracker instance. Creates new one if None.
        """
        self.tracker = tracker if tracker else PredictionTracker()
        self.owns_tracker = tracker is None

    def generate_daily_report(self, target_date: Optional[str] = None) -> str:
        """
        Generate a daily accuracy report.

        Args:
            target_date: Date to report on (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            Formatted report string
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Get predictions for the day
        preds = self.tracker.get_predictions(
            start_date=target_date,
            end_date=target_date,
            only_with_actuals=True
        )

        if preds.empty:
            return f"\n{'='*70}\nDaily Accuracy Report - {target_date}\n{'='*70}\n\nNo predictions with actual results for this date.\n"

        report = []
        report.append("=" * 70)
        report.append(f"Daily Accuracy Report - {target_date}")
        report.append("=" * 70)

        # Overall stats
        total = len(preds)
        mae = (preds['projected_points'] - preds['actual_points']).abs().mean()
        rmse = ((preds['projected_points'] - preds['actual_points']) ** 2).mean() ** 0.5

        report.append(f"\nOverall Performance:")
        report.append(f"  Total predictions: {total}")
        report.append(f"  MAE: {mae:.2f} fantasy points")
        report.append(f"  RMSE: {rmse:.2f} fantasy points")

        # By confidence level
        report.append(f"\nBy Confidence Level:")
        for conf in ['very_high', 'high', 'medium', 'low', 'very_low']:
            conf_preds = preds[preds['confidence'] == conf]
            if not conf_preds.empty:
                conf_mae = (conf_preds['projected_points'] - conf_preds['actual_points']).abs().mean()
                report.append(f"  {conf:12s}: {len(conf_preds):3d} predictions, MAE: {conf_mae:.2f}")

        # By matchup type
        report.append(f"\nBy Matchup Type:")
        for matchup in ['head_to_head', 'pitcher_profile', 'general']:
            m_preds = preds[preds['matchup_type'] == matchup]
            if not m_preds.empty:
                m_mae = (m_preds['projected_points'] - m_preds['actual_points']).abs().mean()
                report.append(f"  {matchup:15s}: {len(m_preds):3d} predictions, MAE: {m_mae:.2f}")

        # Best predictions (closest to actual)
        preds['error'] = (preds['projected_points'] - preds['actual_points']).abs()
        best = preds.nsmallest(5, 'error')

        report.append(f"\nBest Predictions (smallest error):")
        for _, row in best.iterrows():
            report.append(
                f"  {row['player_name']:25s} "
                f"proj: {row['projected_points']:5.1f}  "
                f"actual: {row['actual_points']:5.1f}  "
                f"error: {row['error']:4.1f}"
            )

        # Worst predictions
        worst = preds.nlargest(5, 'error')
        report.append(f"\nWorst Predictions (largest error):")
        for _, row in worst.iterrows():
            report.append(
                f"  {row['player_name']:25s} "
                f"proj: {row['projected_points']:5.1f}  "
                f"actual: {row['actual_points']:5.1f}  "
                f"error: {row['error']:4.1f}"
            )

        report.append("=" * 70)
        return "\n".join(report)

    def generate_weekly_report(self, weeks: int = 1) -> str:
        """
        Generate a weekly accuracy report.

        Args:
            weeks: Number of weeks to look back

        Returns:
            Formatted report string
        """
        days = weeks * 7
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Get predictions
        preds = self.tracker.get_predictions(
            start_date=start_date,
            end_date=end_date,
            only_with_actuals=True
        )

        if preds.empty:
            return f"\n{'='*70}\nWeekly Report - Last {weeks} week(s)\n{'='*70}\n\nNo predictions with actual results in this period.\n"

        report = []
        report.append("=" * 70)
        report.append(f"Weekly Accuracy Report - Last {weeks} week(s)")
        report.append(f"{start_date} to {end_date}")
        report.append("=" * 70)

        # Overall metrics
        accuracy = self.tracker.calculate_accuracy(days=days)
        if accuracy:
            report.append(f"\nOverall Performance:")
            report.append(f"  Sample size: {accuracy['sample_size']}")
            report.append(f"  MAE: {accuracy['mae']:.2f}")
            report.append(f"  RMSE: {accuracy['rmse']:.2f}")
            report.append(f"  Correlation: {accuracy['correlation']:.3f}")
            report.append(f"  Directional accuracy: {accuracy['directional_accuracy']:.1%}")

        # By confidence
        by_conf = self.tracker.calculate_accuracy_by_confidence(days=days)
        if by_conf:
            report.append(f"\nBy Confidence Level:")
            for conf, metrics in by_conf.items():
                report.append(
                    f"  {conf:12s}: MAE {metrics['mae']:.2f}, "
                    f"{metrics['sample_size']:3d} samples"
                )

        # Top performers (lowest MAE)
        player_accuracies = []
        unique_players = preds['player_name'].unique()

        for player in unique_players:
            player_acc = self.tracker.get_player_accuracy(player, days=days)
            if player_acc and player_acc['games'] >= 3:
                player_accuracies.append(player_acc)

        if player_accuracies:
            df_players = pd.DataFrame(player_accuracies)
            df_players = df_players.sort_values('mae')

            report.append(f"\nMost Accurate Predictions (lowest MAE, min 3 games):")
            for _, row in df_players.head(10).iterrows():
                report.append(
                    f"  {row['player_name']:25s} "
                    f"MAE: {row['mae']:4.2f}  "
                    f"games: {row['games']:2d}  "
                    f"bias: {row['bias']:+5.2f}"
                )

            report.append(f"\nLeast Accurate Predictions (highest MAE, min 3 games):")
            for _, row in df_players.tail(10).iterrows():
                report.append(
                    f"  {row['player_name']:25s} "
                    f"MAE: {row['mae']:4.2f}  "
                    f"games: {row['games']:2d}  "
                    f"bias: {row['bias']:+5.2f}"
                )

        report.append("=" * 70)
        return "\n".join(report)

    def generate_player_report(self, player_name: str, days: int = 90) -> str:
        """
        Generate accuracy report for a specific player.

        Args:
            player_name: Player's full name
            days: Number of days to look back

        Returns:
            Formatted report string
        """
        preds = self.tracker.get_predictions(
            player_name=player_name,
            only_with_actuals=True
        )

        if preds.empty:
            return f"\nNo predictions found for {player_name}\n"

        # Calculate accuracy
        accuracy = self.tracker.get_player_accuracy(player_name, days=days)

        report = []
        report.append("=" * 70)
        report.append(f"Player Accuracy Report - {player_name}")
        report.append("=" * 70)

        if accuracy:
            report.append(f"\nOverall Accuracy (last {days} days):")
            report.append(f"  Games tracked: {accuracy['games']}")
            report.append(f"  MAE: {accuracy['mae']:.2f}")
            report.append(f"  Bias: {accuracy['bias']:+.2f} {'(overestimate)' if accuracy['bias'] > 0 else '(underestimate)'}")
            report.append(f"  Mean projected: {accuracy['mean_predicted']:.2f}")
            report.append(f"  Mean actual: {accuracy['mean_actual']:.2f}")
            report.append(f"  Correlation: {accuracy['correlation']:.3f}")

        # Recent games
        preds['error'] = (preds['projected_points'] - preds['actual_points']).abs()
        preds = preds.sort_values('date', ascending=False)

        report.append(f"\nRecent Games:")
        for _, row in preds.head(15).iterrows():
            report.append(
                f"  {row['date']} vs {row['pitcher_faced'] or 'Unknown':20s} "
                f"proj: {row['projected_points']:5.1f}  "
                f"actual: {row['actual_points']:5.1f}  "
                f"error: {row['error']:4.1f}  "
                f"({row['matchup_type']})"
            )

        report.append("=" * 70)
        return "\n".join(report)

    def generate_trend_report(self, days: int = 30) -> str:
        """
        Generate a trend report showing accuracy over time.

        Args:
            days: Number of days to analyze

        Returns:
            Formatted report string
        """
        trend = self.tracker.get_accuracy_trend(days=days)

        if trend.empty:
            return f"\nNo accuracy data available for last {days} days.\n"

        report = []
        report.append("=" * 70)
        report.append(f"Accuracy Trend Report - Last {days} Days")
        report.append("=" * 70)

        report.append(f"\nDate       | Predictions | MAE   | RMSE  | Correlation")
        report.append("-" * 70)

        for _, row in trend.iterrows():
            report.append(
                f"{row['date']} |     {row['predictions_with_actuals']:3d}     | "
                f"{row['mae']:5.2f} | {row['rmse']:5.2f} | {row['correlation']:6.3f}"
            )

        # Overall trend
        if len(trend) > 1:
            recent_mae = trend.head(7)['mae'].mean()
            older_mae = trend.tail(7)['mae'].mean()
            trend_direction = "improving" if recent_mae < older_mae else "worsening"

            report.append("-" * 70)
            report.append(f"\nTrend: MAE is {trend_direction}")
            report.append(f"  Recent 7 days: {recent_mae:.2f}")
            report.append(f"  Previous 7 days: {older_mae:.2f}")
            report.append(f"  Change: {recent_mae - older_mae:+.2f}")

        report.append("=" * 70)
        return "\n".join(report)

    def close(self):
        """Close tracker if we own it."""
        if self.owns_tracker:
            self.tracker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Demo usage
    with AccuracyReporter() as reporter:
        print("\nGenerating Weekly Report...")
        print(reporter.generate_weekly_report(weeks=2))

        print("\nGenerating Trend Report...")
        print(reporter.generate_trend_report(days=14))
