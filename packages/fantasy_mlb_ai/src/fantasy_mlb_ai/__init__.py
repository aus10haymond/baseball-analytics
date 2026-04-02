"""
Fantasy MLB AI - Fantasy Baseball Management and Projections

Integrates with matchup_machine for ML-powered roster recommendations.
Includes pitcher-aware projections, prediction tracking, and automated workflows.
"""

__version__ = "2.0.0"

# Export key classes/functions for convenience
from .ml_projections import MLProjectionEngine
from .pitcher_aware_projections import PitcherAwareEngine
from .prediction_tracker import PredictionTracker
from .accuracy_reporter import AccuracyReporter
from .waiver_wire import WaiverWireAnalyzer

__all__ = [
    "MLProjectionEngine",
    "PitcherAwareEngine",
    "PredictionTracker",
    "AccuracyReporter",
    "WaiverWireAnalyzer",
]
