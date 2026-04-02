"""
Tests for Prediction Tracking System
"""

import pytest
import tempfile
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fantasy_mlb_ai.prediction_tracker import PredictionTracker


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def tracker(temp_db):
    """Create a tracker instance with temporary database."""
    tracker = PredictionTracker(db_path=temp_db)
    yield tracker
    tracker.close()


def test_create_tables(temp_db):
    """Test that database tables are created correctly."""
    tracker = PredictionTracker(db_path=temp_db)

    # Check that tables exist
    tables = tracker.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    table_names = [t[0] for t in tables]
    assert 'predictions' in table_names
    assert 'accuracy_summary' in table_names

    tracker.close()


def test_log_prediction(tracker):
    """Test logging a prediction."""
    today = date.today().strftime('%Y-%m-%d')

    tracker.log_prediction(
        date_str=today,
        player_name="Aaron Judge",
        projected=8.5,
        confidence="high",
        matchup_type="pitcher_profile",
        pitcher="Gerrit Cole",
        team="NYY",
        position="OF"
    )

    # Verify prediction was saved
    preds = tracker.get_predictions(start_date=today, end_date=today)
    assert len(preds) == 1
    assert preds.iloc[0]['player_name'] == "Aaron Judge"
    assert preds.iloc[0]['projected_points'] == 8.5
    assert preds.iloc[0]['confidence'] == "high"


def test_update_actual(tracker):
    """Test updating actual results."""
    today = date.today().strftime('%Y-%m-%d')

    # Log prediction
    tracker.log_prediction(
        date_str=today,
        player_name="Shohei Ohtani",
        projected=7.0,
        confidence="medium"
    )

    # Update with actual
    tracker.update_actual(
        date_str=today,
        player_name="Shohei Ohtani",
        actual_points=9.5,
        pa_actual=4
    )

    # Verify update
    preds = tracker.get_predictions(
        start_date=today,
        end_date=today,
        only_with_actuals=True
    )

    assert len(preds) == 1
    assert preds.iloc[0]['actual_points'] == 9.5
    assert preds.iloc[0]['pa_actual'] == 4


def test_calculate_accuracy(tracker):
    """Test accuracy calculation."""
    today = date.today()

    # Add several predictions with actuals
    test_data = [
        ("Player A", 5.0, 6.0),
        ("Player B", 8.0, 7.5),
        ("Player C", 3.0, 4.0),
        ("Player D", 6.5, 6.0),
        ("Player E", 4.0, 3.5),
    ]

    for i, (player, proj, actual) in enumerate(test_data):
        date_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        tracker.log_prediction(
            date_str=date_str,
            player_name=player,
            projected=proj,
            confidence="medium"
        )
        tracker.update_actual(
            date_str=date_str,
            player_name=player,
            actual_points=actual
        )

    # Calculate accuracy
    accuracy = tracker.calculate_accuracy(days=7)

    assert accuracy is not None
    assert 'mae' in accuracy
    assert 'rmse' in accuracy
    assert 'sample_size' in accuracy
    assert accuracy['sample_size'] == 5

    # MAE should be reasonable
    assert 0 <= accuracy['mae'] <= 2.0


def test_accuracy_by_confidence(tracker):
    """Test accuracy breakdown by confidence level."""
    today = date.today().strftime('%Y-%m-%d')

    # Add predictions with different confidence levels
    test_data = [
        ("Player A", 5.0, 5.2, "high"),
        ("Player B", 6.0, 5.8, "high"),
        ("Player C", 4.0, 6.0, "low"),
        ("Player D", 7.0, 5.0, "low"),
        ("Player E", 5.5, 5.6, "medium"),
    ]

    for player, proj, actual, conf in test_data:
        tracker.log_prediction(
            date_str=today,
            player_name=player,
            projected=proj,
            confidence=conf
        )
        tracker.update_actual(
            date_str=today,
            player_name=player,
            actual_points=actual
        )

    # Get accuracy by confidence
    by_conf = tracker.calculate_accuracy_by_confidence(days=1)

    assert 'high' in by_conf
    assert 'low' in by_conf
    assert 'medium' in by_conf

    # High confidence should have better MAE than low confidence
    # (based on our test data)
    assert by_conf['high']['mae'] < by_conf['low']['mae']


def test_get_player_accuracy(tracker):
    """Test player-specific accuracy calculation."""
    today = date.today()
    player_name = "Test Player"

    # Add multiple games for the player
    for i in range(5):
        date_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        proj = 5.0 + i * 0.5
        actual = proj + 0.3  # Slight overestimate

        tracker.log_prediction(
            date_str=date_str,
            player_name=player_name,
            projected=proj,
            confidence="medium"
        )
        tracker.update_actual(
            date_str=date_str,
            player_name=player_name,
            actual_points=actual
        )

    # Get player accuracy
    player_acc = tracker.get_player_accuracy(player_name, days=7)

    assert player_acc is not None
    assert player_acc['player_name'] == player_name
    assert player_acc['games'] == 5
    assert player_acc['mae'] > 0
    assert player_acc['bias'] < 0  # Should be negative (overestimate)


def test_batch_update_actuals(tracker):
    """Test batch updating of actual results."""
    today = date.today().strftime('%Y-%m-%d')

    # Log multiple predictions
    players = ["Player A", "Player B", "Player C"]
    for player in players:
        tracker.log_prediction(
            date_str=today,
            player_name=player,
            projected=5.0,
            confidence="medium"
        )

    # Batch update actuals
    actuals = [
        {'date': today, 'player_name': 'Player A', 'actual_points': 6.0, 'pa_actual': 4},
        {'date': today, 'player_name': 'Player B', 'actual_points': 4.5, 'pa_actual': 3},
        {'date': today, 'player_name': 'Player C', 'actual_points': 7.0, 'pa_actual': 5},
    ]

    tracker.batch_update_actuals(actuals)

    # Verify all were updated
    preds = tracker.get_predictions(
        start_date=today,
        end_date=today,
        only_with_actuals=True
    )

    assert len(preds) == 3
    assert preds['actual_points'].notna().all()


def test_context_manager(temp_db):
    """Test using tracker as context manager."""
    today = date.today().strftime('%Y-%m-%d')

    with PredictionTracker(db_path=temp_db) as tracker:
        tracker.log_prediction(
            date_str=today,
            player_name="Test Player",
            projected=5.0,
            confidence="medium"
        )

        preds = tracker.get_predictions()
        assert len(preds) == 1

    # Connection should be closed after context
    # Open new connection to verify data persisted
    with PredictionTracker(db_path=temp_db) as tracker:
        preds = tracker.get_predictions()
        assert len(preds) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
