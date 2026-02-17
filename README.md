# Baseball Analytics Monorepo

Unified repository for MLB analytics projects.

## Projects

- **matchup_machine**: Core ML models trained on Statcast data
- **fantasy_mlb_ai**: Fantasy baseball management and projections  
- **diamond_mind**: Multi-agent ML orchestration system

## Setup

```bash
# Install uv (if not already installed)
pip install uv

# Install all packages in development mode
uv pip install -e packages/matchup_machine -e packages/fantasy_mlb_ai -e packages/diamond_mind
```

## Development

See individual package READMEs for details:
- [matchup_machine](packages/matchup_machine/README.md)
- [fantasy_mlb_ai](packages/fantasy_mlb_ai/README.md)
- [diamond_mind](packages/diamond_mind/README.md)
