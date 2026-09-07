# Smart Home Command Parser

A production-style machine learning systems project that trains a Transformer
from scratch to parse natural-language smart-home commands into structured,
validated device actions.

## Example

Input:

```json
{
  "command": "Turn on the kitchen lights"
}
```

Target parsed representation:

```json
{
  "intent": "device_control",
  "action": "turn_on",
  "device": "light",
  "location": "kitchen"
}
```

## Planned stack

- PyTorch Transformer trained from scratch
- FastAPI REST model-serving API
- Docker containerization
- Kubernetes deployment and scaling
- Prometheus metrics and Grafana dashboards
- GitHub Actions CI/CD
- Mock smart-home device controller, followed by optional Home Assistant or MQTT integration

## Local development

```bash
uv sync
make check
make run
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Development roadmap

- [x] Phase 0 — Project foundation and starter FastAPI API
- [ ] Phase 1 — Dataset and Transformer model implementation
- [ ] Phase 2 — Model inference and `/parse-command`
- [ ] Phase 3 — Docker
- [ ] Phase 4 — Kubernetes
- [ ] Phase 5 — Prometheus and Grafana
- [ ] Phase 6 — GitHub Actions CI/CD
- [ ] Phase 7 — Documentation, benchmark, and demo