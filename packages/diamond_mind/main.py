"""
Diamond Mind - Multi-Agent ML System

Main entry point to run all agents.
"""

import asyncio
import signal
from typing import List

from shared import (
    settings,
    ensure_directories,
    init_messaging,
    shutdown_messaging,
    get_agent_logger,
)

from agents.data_quality.agent import DataQualityAgent
# Import other agents as they're implemented
# from agents.orchestrator.agent import OrchestratorAgent
# from agents.model_monitor.agent import ModelMonitorAgent
# from agents.feature_engineer.agent import FeatureEngineerAgent
# from agents.explainer.agent import ExplainerAgent


logger = get_agent_logger("main")


async def run_agents():
    """Run all enabled agents."""
    logger.info("Starting Diamond Mind Multi-Agent System")
    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize messaging
    await init_messaging()
    
    # Create agent instances
    agents = []
    
    if settings.data_quality_enabled:
        agents.append(DataQualityAgent())
        logger.info("Data Quality Agent enabled")
    
    # Add other agents as they're implemented
    # if settings.orchestrator_enabled:
    #     agents.append(OrchestratorAgent())
    # if settings.model_monitor_enabled:
    #     agents.append(ModelMonitorAgent())
    # if settings.feature_engineer_enabled:
    #     agents.append(FeatureEngineerAgent())
    # if settings.explainer_enabled:
    #     agents.append(ExplainerAgent())
    
    if not agents:
        logger.warning("No agents enabled!")
        return
    
    logger.info(f"Starting {len(agents)} agent(s)")
    
    # Start all agents
    tasks = [asyncio.create_task(agent.start()) for agent in agents]
    
    # Wait for all agents
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Received shutdown signal")
    finally:
        # Stop all agents
        logger.info("Stopping agents...")
        for agent in agents:
            await agent.stop()
        
        # Shutdown messaging
        await shutdown_messaging()
        logger.info("Diamond Mind shutdown complete")


def main():
    """Main entry point."""
    print("=" * 70)
    print("DIAMOND MIND - Multi-Agent ML System")
    print("=" * 70)
    print(f"Debug Mode: {settings.debug_mode}")
    print(f"Log Level: {settings.log_level}")
    print(f"Redis: {settings.redis_host}:{settings.redis_port}")
    print("=" * 70)
    print()
    
    # Run the async main function
    try:
        asyncio.run(run_agents())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
