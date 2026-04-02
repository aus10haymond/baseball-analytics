"""
Tests for Pitcher-Aware Projections
"""

import pytest
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fantasy_mlb_ai.pitcher_aware_projections import (
    PitcherAwareEngine,
    compare_projections
)


@pytest.fixture
def engine():
    """Create a pitcher-aware engine instance."""
    return PitcherAwareEngine()


def test_engine_initialization(engine):
    """Test that engine initializes correctly."""
    # Should either load successfully or gracefully handle missing models
    assert hasattr(engine, 'ml_available')

    if engine.ml_available:
        assert hasattr(engine, 'model')
        assert hasattr(engine, 'feature_cols')
        assert hasattr(engine, 'player_index')
        assert hasattr(engine, 'matchups')


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_get_matchup_projection(engine):
    """Test generating a matchup-specific projection."""
    # Use common players that should be in the database
    batter_name = "Aaron Judge"
    pitcher_name = "Gerrit Cole"

    proj = engine.get_matchup_projection(
        batter_name=batter_name,
        pitcher_name=pitcher_name,
        default_pa=4
    )

    # Should return a valid projection dictionary
    assert isinstance(proj, dict)
    assert 'expected_points' in proj
    assert 'confidence' in proj
    assert 'matchup_type' in proj

    # If no error, should have numeric projection
    if proj['error'] is None:
        assert isinstance(proj['expected_points'], (int, float))
        assert proj['expected_points'] >= 0
        assert proj['confidence'] in [
            'very_high', 'high', 'medium', 'low', 'very_low', 'none'
        ]
        assert proj['matchup_type'] in [
            'head_to_head', 'pitcher_profile', 'general'
        ]


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_projection_with_unknown_player(engine):
    """Test projection with unknown player."""
    proj = engine.get_matchup_projection(
        batter_name="Unknown Player XYZ",
        pitcher_name="Gerrit Cole",
        default_pa=4
    )

    # Should handle gracefully with error
    assert proj['expected_points'] is None
    assert proj['error'] is not None
    assert proj['confidence'] == 'none'


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_get_todays_matchups(engine):
    """Test fetching today's matchups from MLB API."""
    matchups = engine.get_todays_matchups()

    # Should return a dictionary (may be empty if off-season)
    assert isinstance(matchups, dict)

    # If there are games today, verify structure
    if matchups:
        for team, info in matchups.items():
            assert 'opponent_pitcher' in info
            assert 'opponent_team' in info
            assert 'is_home' in info
            assert isinstance(info['is_home'], bool)


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_get_roster_matchup_projections(engine):
    """Test generating projections for entire roster."""
    # Create a small test roster
    roster_df = pd.DataFrame({
        'name': ['Aaron Judge', 'Shohei Ohtani', 'Freddie Freeman'],
        'proTeam': ['NYY', 'LAD', 'LAD'],
        'position': ['OF', 'DH', '1B'],
        'lineupSlot': ['OF', 'Util', '1B']
    })

    team_name_map = {
        'NYY': 'New York Yankees',
        'LAD': 'Los Angeles Dodgers'
    }

    result = engine.get_roster_matchup_projections(
        roster_df=roster_df,
        team_name_map=team_name_map,
        default_pa=4
    )

    # Should return a DataFrame with original columns plus projections
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(roster_df)
    assert 'pa_projection' in result.columns
    assert 'pa_confidence' in result.columns
    assert 'pa_matchup_type' in result.columns
    assert 'opponent_pitcher' in result.columns


def test_compare_projections():
    """Test projection comparison function."""
    # Strong positive matchup
    result = compare_projections(5.0, 7.0)
    assert result['advantage'] == 'positive'
    assert result['difference'] == 2.0
    assert result['pct_change'] > 0

    # Weak matchup
    result = compare_projections(6.0, 4.0)
    assert result['advantage'] == 'negative'
    assert result['difference'] == -2.0
    assert result['pct_change'] < 0

    # Neutral matchup
    result = compare_projections(5.0, 5.2)
    assert result['advantage'] == 'neutral'
    assert abs(result['pct_change']) < 5

    # Unknown (missing data)
    result = compare_projections(None, 5.0)
    assert result['advantage'] == 'unknown'
    assert result['difference'] == 0


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_confidence_levels(engine):
    """Test that confidence levels are assigned appropriately."""
    # This is a heuristic test - we can't guarantee specific matchups exist
    # but we can verify the logic is working

    batter_name = "Aaron Judge"
    pitcher_name = "Gerrit Cole"

    proj = engine.get_matchup_projection(batter_name, pitcher_name, default_pa=4)

    if proj['error'] is None:
        # Verify confidence aligns with matchup type and sample size
        matchup_type = proj['matchup_type']
        sample_size = proj.get('sample_size', 0)
        confidence = proj['confidence']

        if matchup_type == 'head_to_head' and sample_size >= 20:
            assert confidence in ['very_high', 'high']
        elif matchup_type == 'general' and sample_size < 100:
            assert confidence in ['low', 'very_low']


@pytest.mark.skipif(
    not PitcherAwareEngine().ml_available,
    reason="ML models not available"
)
def test_outcome_probabilities(engine):
    """Test that outcome probabilities sum to approximately 1.0."""
    batter_name = "Aaron Judge"
    pitcher_name = "Gerrit Cole"

    proj = engine.get_matchup_projection(batter_name, pitcher_name, default_pa=4)

    if proj['outcome_probs'] and proj['error'] is None:
        total_prob = sum(proj['outcome_probs'].values())
        # Should sum to approximately 1.0 (allow small floating point errors)
        assert 0.99 <= total_prob <= 1.01


def test_engine_without_ml():
    """Test that engine handles missing ML gracefully."""
    # This test just verifies the engine doesn't crash if ML unavailable
    engine = PitcherAwareEngine()

    if not engine.ml_available:
        proj = engine.get_matchup_projection("Any Player", "Any Pitcher")
        assert proj['expected_points'] is None
        assert proj['confidence'] == 'none'
        assert 'error' in proj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
