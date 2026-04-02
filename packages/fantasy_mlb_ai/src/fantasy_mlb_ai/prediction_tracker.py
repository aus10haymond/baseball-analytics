"""
Prediction Accuracy Tracking System

Stores daily projections and actual results to measure model accuracy over time.
Provides metrics like MAE, RMSE, and directional accuracy.
"""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PredictionTracker:
    """
    Tracks fantasy baseball predictions and actual results for accuracy analysis.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the prediction tracker.

        Args:
            db_path: Path to SQLite database. Defaults to data/predictions/predictions.db
        """
        if db_path is None:
            # Default to package data directory
            package_dir = Path(__file__).parent.parent.parent
            db_path = package_dir / "data" / "predictions" / "predictions.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                date TEXT NOT NULL,
                player_name TEXT NOT NULL,
                projected_points REAL NOT NULL,
                actual_points REAL,
                confidence TEXT,
                matchup_type TEXT,
                pitcher_faced TEXT,
                team TEXT,
                position TEXT,
                was_started INTEGER,
                pa_actual INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, player_name)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_summary (
                date TEXT PRIMARY KEY,
                total_predictions INTEGER,
                predictions_with_actuals INTEGER,
                mae REAL,
                rmse REAL,
                correlation REAL,
                directional_accuracy REAL,
                high_confidence_mae REAL,
                medium_confidence_mae REAL,
                low_confidence_mae REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    def log_prediction(
        self,
        date_str: str,
        player_name: str,
        projected: float,
        confidence: str = "medium",
        matchup_type: str = "general",
        pitcher: Optional[str] = None,
        team: Optional[str] = None,
        position: Optional[str] = None,
        was_started: bool = True
    ):
        """
        Log a prediction for a player.

        Args:
            date_str: Date in YYYY-MM-DD format
            player_name: Player's full name
            projected: Projected fantasy points
            confidence: Confidence level (high/medium/low)
            matchup_type: Type of matchup (head_to_head/pitcher_profile/general)
            pitcher: Opposing pitcher name
            team: Player's team
            position: Player's position
            was_started: Whether player was in starting lineup
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO predictions
            (date, player_name, projected_points, confidence, matchup_type,
             pitcher_faced, team, position, was_started)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str, player_name, projected, confidence, matchup_type,
            pitcher, team, position, 1 if was_started else 0
        ))
        self.conn.commit()

    def update_actual(
        self,
        date_str: str,
        player_name: str,
        actual_points: float,
        pa_actual: Optional[int] = None
    ):
        """
        Update actual results for a prediction.

        Args:
            date_str: Date in YYYY-MM-DD format
            player_name: Player's full name
            actual_points: Actual fantasy points earned
            pa_actual: Actual plate appearances
        """
        self.conn.execute("""
            UPDATE predictions
            SET actual_points = ?, pa_actual = ?, updated_at = CURRENT_TIMESTAMP
            WHERE date = ? AND player_name = ?
        """, (actual_points, pa_actual, date_str, player_name))
        self.conn.commit()

    def batch_update_actuals(self, actuals: List[Dict]):
        """
        Batch update multiple actual results.

        Args:
            actuals: List of dicts with keys: date, player_name, actual_points, pa_actual
        """
        for actual in actuals:
            self.update_actual(
                date_str=actual['date'],
                player_name=actual['player_name'],
                actual_points=actual['actual_points'],
                pa_actual=actual.get('pa_actual')
            )

    def get_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        player_name: Optional[str] = None,
        only_with_actuals: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve predictions from database.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            player_name: Filter by player name
            only_with_actuals: Only return predictions with actual results

        Returns:
            DataFrame with predictions
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if player_name:
            query += " AND player_name = ?"
            params.append(player_name)

        if only_with_actuals:
            query += " AND actual_points IS NOT NULL"

        query += " ORDER BY date DESC, projected_points DESC"

        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def calculate_accuracy(
        self,
        days: int = 30,
        confidence_level: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Calculate accuracy metrics for recent predictions.

        Args:
            days: Number of days to look back
            confidence_level: Filter by confidence level

        Returns:
            Dictionary with accuracy metrics or None if insufficient data
        """
        query = """
            SELECT projected_points, actual_points, confidence
            FROM predictions
            WHERE actual_points IS NOT NULL
            AND date >= date('now', '-' || ? || ' days')
        """
        params = [days]

        if confidence_level:
            query += " AND confidence = ?"
            params.append(confidence_level)

        rows = self.conn.execute(query, params).fetchall()

        if len(rows) < 5:  # Need at least 5 samples
            return None

        projected = np.array([r[0] for r in rows])
        actual = np.array([r[1] for r in rows])

        # Calculate metrics
        errors = projected - actual
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Correlation
        if len(projected) > 1:
            correlation = float(np.corrcoef(projected, actual)[0, 1])
        else:
            correlation = 0.0

        # Directional accuracy (did we predict up/down correctly?)
        # Compare to previous game's actual or mean
        directional_correct = 0
        directional_total = 0

        for i in range(1, len(rows)):
            if projected[i] > projected[i-1] and actual[i] > actual[i-1]:
                directional_correct += 1
            elif projected[i] < projected[i-1] and actual[i] < actual[i-1]:
                directional_correct += 1
            directional_total += 1

        directional_accuracy = (
            directional_correct / directional_total if directional_total > 0 else 0.0
        )

        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "correlation": round(correlation, 3),
            "directional_accuracy": round(directional_accuracy, 3),
            "sample_size": len(rows),
            "mean_predicted": round(float(np.mean(projected)), 2),
            "mean_actual": round(float(np.mean(actual)), 2),
            "std_predicted": round(float(np.std(projected)), 2),
            "std_actual": round(float(np.std(actual)), 2)
        }

    def calculate_accuracy_by_confidence(self, days: int = 30) -> Dict:
        """
        Calculate accuracy metrics broken down by confidence level.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary mapping confidence level to metrics
        """
        confidence_levels = ['high', 'medium', 'low', 'very_high', 'very_low']
        results = {}

        for level in confidence_levels:
            metrics = self.calculate_accuracy(days=days, confidence_level=level)
            if metrics:
                results[level] = metrics

        return results

    def save_daily_summary(self, target_date: Optional[str] = None):
        """
        Calculate and save daily accuracy summary.

        Args:
            target_date: Date to summarize (defaults to yesterday)
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Get overall metrics
        overall = self.calculate_accuracy(days=1)

        if overall is None:
            print(f"No predictions with actuals for {target_date}")
            return

        # Get by confidence
        by_conf = self.calculate_accuracy_by_confidence(days=1)

        self.conn.execute("""
            INSERT OR REPLACE INTO accuracy_summary
            (date, total_predictions, predictions_with_actuals, mae, rmse,
             correlation, directional_accuracy, high_confidence_mae,
             medium_confidence_mae, low_confidence_mae)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_date,
            overall['sample_size'],
            overall['sample_size'],
            overall['mae'],
            overall['rmse'],
            overall['correlation'],
            overall['directional_accuracy'],
            by_conf.get('high', {}).get('mae'),
            by_conf.get('medium', {}).get('mae'),
            by_conf.get('low', {}).get('mae')
        ))
        self.conn.commit()

    def get_accuracy_trend(self, days: int = 30) -> pd.DataFrame:
        """
        Get accuracy trend over time.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with daily accuracy metrics
        """
        query = """
            SELECT *
            FROM accuracy_summary
            WHERE date >= date('now', '-' || ? || ' days')
            ORDER BY date DESC
        """

        df = pd.read_sql_query(query, self.conn, params=[days])
        return df

    def get_player_accuracy(self, player_name: str, days: int = 90) -> Optional[Dict]:
        """
        Calculate accuracy metrics for a specific player.

        Args:
            player_name: Player's full name
            days: Number of days to look back

        Returns:
            Dictionary with player-specific accuracy metrics
        """
        query = """
            SELECT projected_points, actual_points
            FROM predictions
            WHERE player_name = ?
            AND actual_points IS NOT NULL
            AND date >= date('now', '-' || ? || ' days')
        """

        rows = self.conn.execute(query, (player_name, days)).fetchall()

        if len(rows) < 3:
            return None

        projected = np.array([r[0] for r in rows])
        actual = np.array([r[1] for r in rows])

        errors = projected - actual
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))  # Positive = overestimate

        return {
            "player_name": player_name,
            "games": len(rows),
            "mae": round(mae, 3),
            "bias": round(bias, 3),
            "mean_predicted": round(float(np.mean(projected)), 2),
            "mean_actual": round(float(np.mean(actual)), 2),
            "correlation": round(float(np.corrcoef(projected, actual)[0, 1]), 3) if len(rows) > 1 else 0.0
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Demo usage
    tracker = PredictionTracker()

    # Example: log some predictions
    print("Prediction Tracker Demo")
    print("=" * 60)

    # Show recent predictions
    recent = tracker.get_predictions(only_with_actuals=True)
    if not recent.empty:
        print(f"\nFound {len(recent)} predictions with actual results")
        print(recent[['date', 'player_name', 'projected_points', 'actual_points', 'confidence']].head(10))

        # Calculate accuracy
        accuracy = tracker.calculate_accuracy(days=30)
        if accuracy:
            print("\n30-Day Accuracy Metrics:")
            print(f"  MAE: {accuracy['mae']}")
            print(f"  RMSE: {accuracy['rmse']}")
            print(f"  Correlation: {accuracy['correlation']}")
            print(f"  Sample size: {accuracy['sample_size']}")
    else:
        print("\nNo predictions with actual results yet.")
        print("Run daily workflow to start tracking predictions.")

    tracker.close()
