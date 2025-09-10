# Architecture overview

Components:
- Sensor simulator (src/sensors/simulator.py) - posts JSON to /sensor endpoint
- FastAPI backend (src/app/main.py + src/app/api.py)
  - receives sensor data
  - stores event to SQLite (data/events.db)
  - runs ML detector (src/app/ml/detector.py)
  - uses GPT-OSS adapter (src/app/ml/gpt_oss_adapter.py) for explanation / reasoning (mock by default)
  - triggers RPA engine (src/app/rpa/rpa_engine.py)
- RPA engine
  - toggles devices.json
  - writes logs
  - executes allowed scripts in scripts/
- Dashboard: simple HTML served by FastAPI

Dataflow:
Sensor -> POST /sensor -> store -> ML -> (if anomaly) -> RPA -> log & device toggle -> events endpoint

Security:
- RPA scripts are restricted to the scripts/ folder and executed with limited args.
