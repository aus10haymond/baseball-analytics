"""
Fantasy MLB AI - Fantasy Baseball Management and Projections

Integrates with matchup_machine for ML-powered roster recommendations.
"""

__version__ = "0.1.0"

# Export key classes/functions for convenience
from .ml_projections import MLProjectionEngine

__all__ = [
    "MLProjectionEngine",
]
