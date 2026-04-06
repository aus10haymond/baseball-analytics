"""
Cross-project integrations for diamond_mind.

Provides bridges between diamond_mind agents and the other packages
in the monorepo (matchup_machine and fantasy_mlb_ai).
"""

from .matchup_machine import MatchupMachineIntegration
from .fantasy_mlb_ai import FantasyMLBAIIntegration

__all__ = ["MatchupMachineIntegration", "FantasyMLBAIIntegration"]
