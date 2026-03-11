# Diamond Mind ⚾🧠

**Multi-agent ML system for baseball analytics with autonomous monitoring, retraining, and feature engineering**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

Diamond Mind is a production-ready multi-agent ML platform that autonomously manages the entire ML lifecycle for baseball analytics:

- 🔍 **Data Quality Agent** - Detects anomalies, validates schemas, repairs data issues
- 📊 **Model Monitor Agent** - Tracks drift, triggers retraining, manages A/B testing
- 🧬 **Feature Engineer Agent** - Discovers new features via genetic algorithms + LLM
- 💡 **Explainer Agent** - Generates human-readable explanations (SHAP + LLM)
- 🎯 **Orchestrator Agent** - LLM-powered coordination and decision-making

## 🏗️ Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DIAMOND MIND                             │
│               Multi-Agent ML System                             │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │  Orchestrator Agent  │
                    │   (LLM Coordinator)  │
                    └──────────┬───────────┘
                               │
                   ┌───────────┼───────────┐
                   │   Redis Message Queue │
                   │   (Task Distribution) │
                   └───────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         │                     │                     │
  ┌──────▼─────┐        ┌─────▼──────┐       ┌─────▼──────┐
  │    Data    │        │   Model    │       │  Feature   │
  │  Quality   │        │  Monitor   │       │  Engineer  │
  │   Agent    │        │   Agent    │       │   Agent    │
  └──────┬─────┘        └─────┬──────┘       └─────┬──────┘
         │                     │                     │
         │                     │                     │
         │              ┌──────▼──────┐             │
         │              │  Explainer  │             │
         │              │    Agent    │             │
         │              └──────┬──────┘             │
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Data Layer        │
                    ├──────────────────────┤
                    │  Sister Projects:    │
                    │  • matchup_machine   │
                    │  • fantasy_mlb_ai    │
                    └──────────────────────┘
```

### Message Flow

```
1. Task Creation
   User/Orchestrator → Task → Redis Queue

2. Task Execution
   Agent ← Task ← Redis Queue
   Agent → Process → Result
   Agent → Result → Redis Queue

3. Alert Flow
   Agent → Alert → Redis Pub/Sub → All Subscribers

4. Cross-Agent Communication
   Agent A → Task (for Agent B) → Redis → Agent B
   Agent B → Result → Redis → Agent A
```

## ✨ Features

### Infrastructure
- **Async messaging** via Redis for agent communication
- **Type-safe schemas** using Pydantic for all messages
- **Structured logging** with JSON formatting
- **Configuration management** via environment variables
- **Base agent framework** with lifecycle management, error handling, heartbeats

### Agents

#### Data Quality Agent
- Anomaly detection (Isolation Forest, statistical methods)
- Schema validation with auto-caching
- Missing value detection and imputation
- Outlier detection (IQR, Z-score)
- Automatic data repair

#### Model Monitor Agent
- Concept drift detection (PSI, KS tests)
- Performance tracking (accuracy, AUC, calibration)
- Automatic retraining triggers
- A/B testing framework
- Model versioning and rollback

#### Feature Engineer Agent
- Genetic algorithm for feature discovery
- LLM-based feature suggestions
- Cross-validation for feature evaluation
- Multicollinearity detection (VIF)
- Feature importance analysis

#### Explainer Agent
- SHAP integration for feature attribution
- LLM-powered narrative generation
- Counterfactual "what if" scenarios
- Explanation caching

#### Orchestrator Agent
- LLM-based task routing
- Conflict resolution between agents
- System health monitoring
- Priority management
- Intelligent decision-making

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Redis (local or Docker)
- Sister projects: `matchup_machine`, `fantasy_mlb_ai` (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diamond-mind.git
cd diamond-mind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Setup Redis

```bash
# Using Docker (recommended)
docker run -d -p 6379:6379 redis:latest

# Or install Redis locally
# Windows: https://github.com/microsoftarchive/redis/releases
# Mac: brew install redis
# Linux: sudo apt-get install redis-server
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Minimum required:
# - DM_REDIS_HOST=localhost
# - DM_REDIS_PORT=6379
# - DM_LLM_API_KEY=your_key (for Orchestrator/Explainer)
```

### Run

```bash
# Run all enabled agents
python main.py

# Run single agent for testing
python agents/data_quality/agent.py
```

## 📖 Documentation

- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Complete overview and setup guide
- **[API Reference](.codebase_reference.md)** - Detailed API documentation (internal)
- **[Implementation Plan](PLAN.md)** - Step-by-step implementation roadmap (internal)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=shared --cov=agents --cov-report=html

# Run specific test
pytest tests/unit/test_schemas.py -v
```

## 🛠️ Development

### Project Structure

```
diamond_mind/
├── shared/              # Core infrastructure (✅ Complete)
│   ├── schemas.py       # Pydantic models
│   ├── config.py        # Configuration
│   ├── messaging.py     # Redis messaging
│   ├── logging_utils.py # Logging
│   └── base_agent.py    # Base agent class
│
├── agents/              # Agent implementations (🚧 Stubs)
│   ├── data_quality/
│   ├── model_monitor/
│   ├── feature_engineer/
│   ├── explainer/
│   └── orchestrator/
│
├── integrations/        # Sister project bridges
├── tests/              # Test suite
├── infrastructure/     # Docker, configs
└── main.py            # Entry point
```

### Adding a New Agent

1. Create agent directory: `agents/my_agent/`
2. Implement `agent.py` following the stub pattern
3. Add to `AgentType` enum in `shared/schemas.py`
4. Enable in `.env`: `DM_MY_AGENT_ENABLED=true`
5. Import in `main.py`
6. Write tests in `tests/unit/agents/test_my_agent.py`

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy shared/ agents/
```

## 🔧 Configuration

All configuration via environment variables with `DM_` prefix:

```bash
# Redis
DM_REDIS_HOST=localhost
DM_REDIS_PORT=6379

# Agents (enable/disable)
DM_DATA_QUALITY_ENABLED=true
DM_MODEL_MONITOR_ENABLED=true
DM_FEATURE_ENGINEER_ENABLED=true
DM_EXPLAINER_ENABLED=true
DM_ORCHESTRATOR_ENABLED=true

# LLM (for Orchestrator & Explainer)
DM_LLM_PROVIDER=openai
DM_LLM_MODEL=gpt-4
DM_LLM_API_KEY=sk-...

# Paths to sister projects (optional)
DM_FANTASY_MLB_PATH=/path/to/fantasy_mlb_ai
DM_MATCHUP_MACHINE_PATH=/path/to/matchup_machine
```

See [.env.example](.env.example) for all options.

## 🤝 Integration with Sister Projects

### matchup_machine
**Purpose**: Provides XGBoost models and 2M+ Statcast pitch data

**Resources Used**:
- `models/xgb_outcome_model.joblib` - Trained model
- `data/matchups.parquet` - Historical data
- `data/player_index.csv` - Player mappings

### fantasy_mlb_ai
**Purpose**: Fantasy baseball management and projections

**Resources Used**:
- `ml_projections.py` - Projection engine
- `data/rosters_2026.csv` - Current rosters
- `data/draft_rankings/` - Draft rankings

## 📊 Monitoring

### Agent Health

Each agent exposes:
- Heartbeat every 60s
- Task completion metrics
- Error rates
- Uptime tracking

### System Metrics

```python
from shared.messaging import message_queue

# Check queue depth
depth = await message_queue.get_queue_depth("diamond_mind:tasks")

# Check agent heartbeat
last_seen = await message_queue.get_agent_heartbeat("data_quality")
```

## 🐛 Troubleshooting

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping  # Should return PONG

# Check connection settings in .env
DM_REDIS_HOST=localhost
DM_REDIS_PORT=6379
```

### Agent Not Starting
```bash
# Check logs
tail -f logs/agent_name/agent_name.log

# Check if agent is enabled
DM_AGENT_NAME_ENABLED=true  # In .env
```

### Import Errors
```bash
# Ensure sister projects are accessible
# Either set paths in .env or ensure they're in ../
```

## 📝 License

MIT License - see [LICENSE](LICENSE)

**Data Usage Disclaimer**: This project uses MLB Stats API and Statcast data for educational purposes. Commercial use requires compliance with MLB data terms of service.

## 👤 Author

Austen Haymond

## 🙏 Acknowledgments

- **matchup_machine** - XGBoost models and Statcast data
- **fantasy_mlb_ai** - Fantasy projections and roster management
- Baseball community for analytics inspiration

---

**Built with**: Python, Redis, Pydantic, XGBoost, AsyncIO  
