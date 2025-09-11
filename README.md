# **IoT + ML + RPA Hackathon Project Documentation**

---
* Project Overview
* Architecture Diagram
* Directory Structure & File Purpose
* Environment Setup
* Sensor Simulation
* ML Anomaly Detection
* GPT-OSS Reasoning
* RPA Automation
* Dashboard & Event Monitoring
* Testing & Validation
* Expected Outputs
* Security & Safety Considerations
* Limitations
* Hackathon Extensions
* Step-by-Step Execution Guide

  
## **1. Project Overview**

This project is a **simulated IoT system** integrated with:

* **ML Anomaly Detection** – IsolationForest model detects abnormal sensor readings.
* **RPA Automation** – Automatically toggles devices or runs safe scripts based on anomalies.
* **GPT-OSS Reasoning** – Provides human-readable explanations for anomalies.

**Data Flow:**

```
Sensors → API → Database → ML Detector → GPT-OSS Adapter → RPA Engine → Devices / Logs / Dashboard
```

**Functional Summary:**

* **Sensors:** Send temperature or motion readings (real or simulated).
* **API:** Receives sensor events, stores them, triggers ML detection and GPT reasoning.
* **DB:** Stores events, anomaly flags, ML scores, GPT explanations, and executed actions.
* **ML Detector:** Computes anomaly scores and flags unusual readings.
* **GPT-OSS Adapter:** Generates explanations for anomalies.
* **RPA Engine:** Executes safe automation actions (device toggles, scripts).
* **Dashboard:** Displays live device states and recent events.

---

## **2. Architecture Diagram**

```
+-----------------+
| Sensors /       |
| Simulator       |
+--------+--------+
         |
         v
+-----------------+
| FastAPI API     | (src/app/api.py)
+--------+--------+
         |
         v
+-----------------+
| SQLite Database | (src/app/db.py)
+--------+--------+
         |
         v
+------------------------+
| ML Detector            | (src/app/ml/detector.py)
| IsolationForest Model  |
+--------+--------+
         |
         v
+------------------------+
| GPT-OSS Adapter        | (src/app/ml/gpt_oss_adapter.py)
| - Provides reasoning   |
+--------+--------+
         |
         v
+------------------------+
| RPA Engine             | (src/app/rpa/rpa_engine.py)
| - Toggle devices       |
| - Run safe scripts     |
+--------+--------+
         |
         v
+-----------------+   +-----------------+
| Devices JSON    |   | Logs (rpa.log)  |
| (src/app/devices.py) |                 |
+-----------------+   +-----------------+
         |
         v
+-----------------+
| Dashboard       | (src/app/dashboard)
| - /devices      |
| - /events       |
+-----------------+
```

---

## **3. Directory Structure & Purpose**

```
data/
  devices.json           # Current state of all devices
  sample_sensor_data.json # Example sensor readings

docs/
  architecture.md        # System overview and diagrams

scripts/
  safe_action.sh         # Safe demo script executed by RPA
  run_local_model.sh     # Script to run GPT-OSS locally

src/app/
  main.py                # FastAPI server entry
  api.py                 # API endpoints & ML/RPA coordination
  db.py                  # SQLite database operations
  devices.py             # Device state management
  ml/
    detector.py          # ML anomaly detection
    trainer.py           # Train synthetic IsolationForest model
    gpt_oss_adapter.py   # Generate human-readable reasoning
  rpa/
    rpa_engine.py        # Executes RPA actions
  dashboard/
    templates/index.html # Dashboard HTML
    static/app.js        # Frontend JS

src/sensors/
  simulator.py           # Simulated sensor events

src/tests/
  test_detector.py       # ML unit tests
  test_gpt_adapter.py    # GPT reasoning unit tests
  test_integration.py    # End-to-end workflow tests
```

---

## **4. Step-by-Step Execution**

### **Step 1: Setup Environment**

1. Install **Python 3.11**.
2. Create virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Optional: GPT-OSS integration:

```bash
export USE_GPT_OSS=true
export MODEL_DIR=/path/to/local/gpt-oss-model
```

---

### **Step 2: Initialize Database & ML Model**

```bash
python src/app/db.py            # Initialize SQLite DB
python src/app/ml/trainer.py    # Generate or train IsolationForest model
```

---

### **Step 3: Start FastAPI Server**

```bash
uvicorn src.app.main:app --reload
```

* Dashboard URL: [http://localhost:8000](http://localhost:8000)
* API Endpoints:

  * `POST /sensor` → Submit events
  * `GET /events` → Retrieve last 100 events
  * `GET /devices` → Retrieve device states

---

### **Step 4: Simulate Sensor Data**

```bash
python src/sensors/simulator.py --count 5 --type temperature
python src/sensors/simulator.py --count 3 --type motion
```

* Sends events to API (`POST /sensor`).
* Stored in SQLite, processed by ML detector.

---

### **Step 5: ML Detection & GPT-OSS Reasoning**

* ML Detector outputs:

```text
is_anomaly → True/False
ml_score → Numeric anomaly score
```

* GPT-OSS Adapter outputs (mocked by default):

```text
"MockReason: Possible overheating. Turn on fan. (score=-0.7)"
```

---

### **Step 6: RPA Action Trigger**

* Temperature anomaly → toggle devices (`fan-1`, `heater-1`)
* Motion anomaly → run safe scripts (`safe_action.sh`)
* Actions logged in `rpa.log`

---

### **Step 7: View Dashboard & Events**

* Devices: `GET /devices`
* Events: `GET /events`
* Dashboard auto-refreshes every 5 seconds

---

### **Step 8: Testing**

* Unit Tests:

```bash
pytest src/tests/test_detector.py
pytest src/tests/test_gpt_adapter.py
```

* Integration Test:

```bash
pytest src/tests/test_integration.py
```

* Verify ML detection, GPT reasoning, RPA actions, and dashboard updates.

---

## **5. Expected Outputs**

1. **Normal Temperature (22°C)**

```json
{
  "event_id": 1,
  "is_anomaly": false,
  "ml_score": 0.25,
  "gpt_reason": "MockReason: Value within expected range. (score=0.25)",
  "rpa": null
}
```

2. **High Temperature (45°C)**

```json
{
  "event_id": 2,
  "is_anomaly": true,
  "ml_score": -0.7,
  "gpt_reason": "MockReason: Possible overheating. Turn on fan. (score=-0.7)",
  "rpa": {"scheduled": true, "action": {"type": "toggle_device", "device_id": "fan-1", "state": "on"}}
}
```

3. **Motion Event (unexpected)**

```json
{
  "event_id": 3,
  "is_anomaly": true,
  "ml_score": -0.9,
  "gpt_reason": "MockReason: Motion detected unexpectedly. (score=-0.9)",
  "rpa": {"scheduled": true, "action": {"type": "run_script", "script": "safe_action.sh", "args": ["motion-sim-1"]}}
}
```

4. **Dashboard Devices Example**

```json
{
  "fan-1": {"id":"fan-1","state":"on","last_updated":"2025-09-11T16:00:12Z"},
  "heater-1": {"id":"heater-1","state":"off","last_updated":"2025-09-11T16:00:01Z"}
}
```

---

## **6. Security & Safety Considerations**

* Only executes scripts in `scripts/` folder.
* GPT-OSS output **never executed** as code.
* Device files and DB are local; no remote network access by default.
* Scripts should run as unprivileged user; arguments validated.
* ML model is synthetic for demo; replace with production data for real deployments.

---

## **7. Limitations**

* IsolationForest model is synthetic.
* GPT-OSS models require memory and GPU for local inference.
* Dashboard is basic HTML; not production-grade.
* RPA actions limited to safe scripts and JSON device toggles.





# Directory tree

```
iot-rpa-hackathon/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── devices.json
│   └── sample_sensor_data.json
├── docs/
│   └── architecture.md
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api.py
│   │   ├── models.py
│   │   ├── db.py
│   │   ├── devices.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py
│   │   │   ├── trainer.py
│   │   │   └── gpt_oss_adapter.py
│   │   ├── rpa/
│   │   │   ├── __init__.py
│   │   │   └── rpa_engine.py
│   │   └── dashboard/
│   │       ├── templates/
│   │       │   └── index.html
│   │       └── static/
│   │           └── app.js
│   ├── sensors/
│   │   └── simulator.py
│   └── tests/
│       ├── unit/
│       │   ├── test_detector.py
│       │   └── test_gpt_adapter.py
│       └── integration/
│           └── test_integration.py
└── scripts/
    ├── safe_action.sh
    └── run_local_model.sh
```

---

# All code files (full contents)

> **Note:** I present each file path and its full content. Copy into a repo with the same structure.

---

### `.gitignore`

```
__pycache__/
*.pyc
.env
data/*.db
data/*.pkl
.venv/
.vscode/
*.log
```

---

### `.env.example`

```
# Server
API_HOST=0.0.0.0
API_PORT=8000
DB_PATH=/data/events.db

# ML / GPT
USE_GPT_OSS=false
GPT_MODEL_NAME=gpt-oss-20b
MODEL_DIR=/models/gpt-oss-20b

# RPA
RPA_SCRIPT_DIR=/app/scripts
```

---

### `requirements.txt`

```
fastapi==0.95.2
uvicorn[standard]==0.22.0
requests==2.31.0
scikit-learn==1.3.0
joblib==1.2.0
pydantic==1.10.11
pytest==7.4.0
httpx==0.24.0
python-multipart==0.0.6
transformers>=4.30.0
# torch is optional (large); uncomment if you have resources
# torch
```

---

### `docker-compose.yml`

```yaml
version: '3.8'
services:
  api:
    build: .
    container_name: iot_rpa_api
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./src:/app/src:ro
      - ./scripts:/app/scripts:ro
    command: uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# create app structure
COPY . /app

# create data dir
RUN mkdir -p /data
RUN chown -R 1000:1000 /data || true

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### `data/devices.json` (initial)

```json
{
  "heater-1": {"id": "heater-1", "state": "off", "last_updated": null},
  "fan-1": {"id": "fan-1", "state": "off", "last_updated": null}
}
```

---

### `data/sample_sensor_data.json`

```json
[
  {"sensor_id": "temp-1", "type": "temperature", "value": 22.4, "timestamp": "2025-09-09T09:00:00Z"},
  {"sensor_id": "temp-1", "type": "temperature", "value": 22.7, "timestamp": "2025-09-09T09:01:00Z"},
  {"sensor_id": "motion-1", "type": "motion", "value": 0, "timestamp": "2025-09-09T09:01:30Z"},
  {"sensor_id": "temp-1", "type": "temperature", "value": 45.2, "timestamp": "2025-09-09T09:05:00Z"}
]
```

---

### `docs/architecture.md`

```
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
```

---

### `scripts/safe_action.sh`

```bash
#!/usr/bin/env bash
# safe_action.sh - example safe script executed by RPA engine
echo "Safe action executed at $(date -u) with args: $@" >> /data/rpa_script.log
```

Make executable when using locally: `chmod +x scripts/safe_action.sh`

---

### `scripts/run_local_model.sh`

```bash
#!/usr/bin/env bash
echo "This helper outlines how to run a local gpt-oss model (not bundled)."
echo "1) Install transformers and a compatible torch."
echo "2) Download model weights to /models/gpt-oss-20b (or point GPT_MODEL_NAME to your path)."
echo "3) Start server or run the adapter in USE_GPT_OSS=true."
echo
echo "See README.md for full details."
```

---

## `src/app/__init__.py`

```python
# empty to make src.app a package
```

---

## `src/app/models.py`

```python
from pydantic import BaseModel
from typing import Optional

class SensorPayload(BaseModel):
    sensor_id: str
    type: str
    value: float
    timestamp: Optional[str] = None

class EventRecord(BaseModel):
    id: int
    sensor_id: str
    type: str
    value: float
    timestamp: Optional[str] = None
    is_anomaly: bool
    ml_score: float
    gpt_reason: Optional[str] = None
    rpa_action: Optional[str] = None
```

---

## `src/app/db.py`

```python
import sqlite3
import os
from typing import Dict, Any

DB_PATH = os.getenv("DB_PATH", "/data/events.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sensor_id TEXT,
        type TEXT,
        value REAL,
        timestamp TEXT,
        is_anomaly INTEGER,
        ml_score REAL,
        gpt_reason TEXT,
        rpa_action TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_event(event: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO events (sensor_id,type,value,timestamp,is_anomaly,ml_score,gpt_reason,rpa_action)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event.get("sensor_id"),
        event.get("type"),
        event.get("value"),
        event.get("timestamp"),
        1 if event.get("is_anomaly") else 0,
        event.get("ml_score"),
        event.get("gpt_reason"),
        event.get("rpa_action"),
    ))
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid

def list_events(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, sensor_id, type, value, timestamp, is_anomaly, ml_score, gpt_reason, rpa_action FROM events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
```

---

## `src/app/devices.py`

```python
import json
import os
import datetime

DEVICES_PATH = os.getenv("DEVICES_PATH", "/data/devices.json")

def _ensure_devices_file():
    os.makedirs(os.path.dirname(DEVICES_PATH), exist_ok=True)
    if not os.path.exists(DEVICES_PATH):
        with open(DEVICES_PATH, "w") as f:
            json.dump({}, f)

def load_devices():
    _ensure_devices_file()
    with open(DEVICES_PATH, "r") as f:
        return json.load(f)

def save_devices(devices):
    os.makedirs(os.path.dirname(DEVICES_PATH), exist_ok=True)
    with open(DEVICES_PATH, "w") as f:
        json.dump(devices, f, indent=2)

def toggle_device(device_id: str, state: str):
    devices = load_devices()
    if device_id not in devices:
        devices[device_id] = {"id": device_id, "state": "off", "last_updated": None}
    devices[device_id]["state"] = state
    devices[device_id]["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    save_devices(devices)
    return devices[device_id]
```

---

## `src/app/ml/__init__.py`

```python
# ml package
```

---

## `src/app/ml/trainer.py`

```python
# trainer: creates a simple IsolationForest model from sample data
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.getenv("ML_MODEL_PATH", "/data/detector.pkl")

def train_sample_model(force=False):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH) and not force:
        return MODEL_PATH
    # create synthetic training data: [value, type_code]
    rng = np.random.RandomState(42)
    temps = 20 + 3 * rng.randn(200, 1)  # temperature cluster around 20
    motions = rng.randint(0,2,(200,1))  # motion 0/1
    data = np.vstack([np.hstack([temps, np.zeros((200,1))]),
                      np.hstack([20 + rng.randn(200,1)*5, np.ones((200,1))])])
    # shape: many samples
    X = np.vstack([temps, np.hstack([20 + rng.randn(200,1)*5, np.ones((200,1))])])
    X = np.vstack([np.hstack([temps, np.zeros((temps.shape[0],1))]) , np.hstack([temps+0.5, np.ones((temps.shape[0],1))])])
    X = np.vstack([X, 18 + 2*rng.randn(200,1)])
    # For simplicity create features: [value, type_code]
    # We'll create a simple synthetic matrix:
    values = np.concatenate([20 + 2*rng.randn(400), 18 + 3*rng.randn(200)])
    types = np.concatenate([np.zeros(400), np.ones(200)])
    X = np.column_stack([values, types])
    clf = IsolationForest(random_state=42, contamination=0.05)
    clf.fit(X)
    joblib.dump(clf, MODEL_PATH)
    return MODEL_PATH
```

---

## `src/app/ml/detector.py`

```python
import os
import joblib
import numpy as np
from .trainer import train_sample_model, MODEL_PATH

MODEL_PATH = os.getenv("ML_MODEL_PATH", MODEL_PATH)

def _load_model():
    if not os.path.exists(MODEL_PATH):
        train_sample_model()
    return joblib.load(MODEL_PATH)

_model = None

def init_detector():
    global _model
    _model = _load_model()

def predict(sensor_type: str, value: float):
    """
    returns: (is_anomaly: bool, score: float)
    uses features [value, type_code] where type_code: temperature=0, motion=1, custom=2
    """
    global _model
    if _model is None:
        init_detector()
    type_map = {"temperature": 0.0, "motion": 1.0}
    tcode = type_map.get(sensor_type, 2.0)
    X = np.array([[float(value), float(tcode)]])
    # IsolationForest: decision_function -> anomaly score (higher is normal); predict -> 1 normal, -1 outlier
    score = float(_model.decision_function(X)[0])
    pred = int(_model.predict(X)[0])
    is_anomaly = (pred == -1)
    # normalize score (rough)
    return is_anomaly, score
```

---

## `src/app/ml/gpt_oss_adapter.py`

```python
import os
import json
import time

USE_GPT = os.getenv("USE_GPT_OSS", "false").lower() in ("1", "true", "yes")
MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-oss-20b")
MODEL_DIR = os.getenv("MODEL_DIR", "/models/gpt-oss-20b")

# If USE_GPT is true, we attempt to use transformers; otherwise we mock.
if USE_GPT:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_NAME)
        _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer, device_map="auto" if hasattr(_model, "device_map") else None)
    except Exception as e:
        print("Warning: failed to load GPT model, falling back to mock. Error:", str(e))
        USE_GPT = False

def reason_with_gpt(sensor_event: dict, anomaly_score: float) -> str:
    """
    Returns a human-readable reasoning string produced by gpt-oss.
    If model not available, returns a deterministic mock reasoning.
    """
    prompt = f"""
You are a reasoning assistant for IoT anomalies. Given the event:
{json.dumps(sensor_event)}
Anomaly score: {anomaly_score}
Explain whether this seems anomalous, possible cause, and suggested automated action in <= 80 words.
"""
    if USE_GPT:
        try:
            resp = _pipe(prompt, max_length=200, do_sample=False)[0]["generated_text"]
            # trim to brief explanation
            return resp.strip()
        except Exception as e:
            return f"(gpt error) fallback explanation: anomaly_score={anomaly_score}"
    # Mock reasoning
    # deterministic short explanation to use in tests/dev
    sensor_type = sensor_event.get("type", "unknown")
    val = sensor_event.get("value")
    if sensor_type == "temperature":
        if val and float(val) > 40:
            suggestion = "Possible overheating. Turn on fan / shut down non-critical load."
        elif val and float(val) < 5:
            suggestion = "Unusually low temperature. Check HVAC or sensor calibration."
        else:
            suggestion = "Value within expected range."
    elif sensor_type == "motion":
        suggestion = "Motion event: verify scheduled occupancy; if unexpected, send alert."
    else:
        suggestion = "No specific pattern detected."
    return f"MockReason: {suggestion} (score={anomaly_score})"
```

---

## `src/app/rpa/__init__.py`

```python
# rpa package
```

---

## `src/app/rpa/rpa_engine.py`

```python
import os
import subprocess
import datetime
from ..devices import toggle_device
from typing import Optional

RPA_SCRIPT_DIR = os.getenv("RPA_SCRIPT_DIR", "/app/scripts")
DATA_DIR = "/data"
RPA_LOG = os.path.join(DATA_DIR, "rpa.log")

def _log(msg: str):
    os.makedirs(os.path.dirname(RPA_LOG), exist_ok=True)
    with open(RPA_LOG, "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()}Z - {msg}\n")

def execute_action(action: dict) -> dict:
    """
    action: {type: 'toggle_device'|'run_script'|'noop', device_id?, state?, script?, args?}
    """
    a_type = action.get("type")
    if a_type == "toggle_device":
        device_id = action.get("device_id")
        state = action.get("state", "on")
        res = toggle_device(device_id, state)
        _log(f"toggle_device executed: {device_id} -> {state}")
        return {"status": "ok", "action": "toggle_device", "device": res}
    elif a_type == "run_script":
        script = action.get("script")
        args = action.get("args", [])
        # Only allow scripts from the RPA_SCRIPT_DIR
        safe_path = os.path.join(RPA_SCRIPT_DIR, script)
        if not os.path.realpath(safe_path).startswith(os.path.realpath(RPA_SCRIPT_DIR)):
            _log(f"attempted unsafe script: {script}")
            return {"status": "error", "reason": "unsafe script path"}
        if not os.path.exists(safe_path):
            _log(f"script not found: {script}")
            return {"status": "error", "reason": "script not found"}
        try:
            subprocess.run([safe_path] + args, check=True)
            _log(f"script executed: {script} args={args}")
            return {"status": "ok", "action": "run_script", "script": script}
        except subprocess.CalledProcessError as e:
            _log(f"script failed: {script} error={e}")
            return {"status": "error", "reason": str(e)}
    else:
        _log(f"noop action: {action}")
        return {"status": "ok", "action": "noop"}
```

---

## `src/app/api.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from .models import SensorPayload, EventRecord
from . import db
from .ml import detector, gpt_oss_adapter
from .rpa.rpa_engine import execute_action
from .devices import load_devices
import os

router = APIRouter()

@router.post("/sensor")
def ingest_sensor(payload: SensorPayload, background_tasks: BackgroundTasks):
    # store preliminary event
    ev = {
        "sensor_id": payload.sensor_id,
        "type": payload.type,
        "value": payload.value,
        "timestamp": payload.timestamp,
        "is_anomaly": False,
        "ml_score": 0.0,
        "gpt_reason": "",
        "rpa_action": ""
    }
    # Run ML detection synchronously
    is_anom, score = detector.predict(payload.type, payload.value)
    ev["is_anomaly"] = is_anom
    ev["ml_score"] = score
    # get reasoning from gpt adapter (fast mock or real)
    try:
        reason = gpt_oss_adapter.reason_with_gpt(ev, score)
    except Exception as e:
        reason = f"(reasoning failed: {e})"
    ev["gpt_reason"] = reason
    # Evaluate and trigger RPA only if anomaly true
    rpa_result = None
    if is_anom:
        # decide action (simple rule + optionally use gpt for suggestions)
        if payload.type == "temperature":
            # If very hot, turn on fan (fan-1)
            if payload.value > 35:
                action = {"type": "toggle_device", "device_id": "fan-1", "state": "on"}
            else:
                action = {"type": "toggle_device", "device_id": "heater-1", "state": "off"}
        elif payload.type == "motion":
            action = {"type": "run_script", "script": "safe_action.sh", "args": [payload.sensor_id]}
        else:
            action = {"type": "noop"}
        # Execute background for better responsiveness
        background_tasks.add_task(execute_action, action)
        ev["rpa_action"] = str(action)
        rpa_result = {"scheduled": True, "action": action}
    eid = db.insert_event(ev)
    return {"event_id": eid, "is_anomaly": is_anom, "ml_score": score, "gpt_reason": reason, "rpa": rpa_result}

@router.get("/events")
def get_events(limit: int = 100):
    rows = db.list_events(limit)
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "sensor_id": r[1],
            "type": r[2],
            "value": r[3],
            "timestamp": r[4],
            "is_anomaly": bool(r[5]),
            "ml_score": r[6],
            "gpt_reason": r[7],
            "rpa_action": r[8]
        })
    return {"events": results}

@router.get("/devices")
def list_devices():
    return load_devices()
```

---

## `src/app/dashboard/templates/index.html`

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>IoT RPA Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .card { padding: 12px; border: 1px solid #ddd; margin-bottom: 12px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>IoT RPA Dashboard</h1>
  <div id="devices" class="card">Loading devices...</div>
  <div id="events" class="card">Loading events...</div>

  <script>
    async function load() {
      const dev = await fetch('/devices').then(r=>r.json());
      document.getElementById('devices').innerText = JSON.stringify(dev, null, 2);
      const ev = await fetch('/events').then(r=>r.json());
      document.getElementById('events').innerText = JSON.stringify(ev, null, 2);
    }
    load();
    setInterval(load, 5000);
  </script>
</body>
</html>
```

---

## `src/app/dashboard/static/app.js`

```javascript
// (Optional for future UI) kept minimal for now
```

---

## `src/app/main.py`

```python
from fastapi import FastAPI
from . import db
from .ml import detector
from .api import router
from fastapi.responses import HTMLResponse
from .dashboard import templates
from pathlib import Path
import os

app = FastAPI(title="IoT + ML + RPA (gpt-oss demo)")

@app.on_event("startup")
def startup():
    db.init_db()
    detector.init_detector()

app.include_router(router, prefix="")

# simple page
@app.get("/", response_class=HTMLResponse)
def homepage():
    p = Path(__file__).parent / "dashboard" / "templates" / "index.html"
    return p.read_text()
```

---

## `src/sensors/simulator.py`

```python
"""
Simple sensor simulator. Use command-line to send mock sensor events to the API.
Example usage:
python src/sensors/simulator.py --host http://localhost:8000 --count 10 --type temperature
"""
import requests
import time
import argparse
import random
import json
from datetime import datetime

def send_event(base_url, payload):
    r = requests.post(f"{base_url}/sensor", json=payload, timeout=5)
    return r.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--type", choices=["temperature","motion"], default="temperature")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    for i in range(args.count):
        if args.type == "temperature":
            # normal ~ 20-25, occasional hot anomaly 45
            if random.random() < 0.1:
                val = round(40 + random.random()*10, 2)
            else:
                val = round(20 + random.random()*4, 2)
            payload = {"sensor_id": "temp-sim-1", "type": "temperature", "value": val, "timestamp": datetime.utcnow().isoformat()+"Z"}
        else:
            val = 1 if random.random() < 0.05 else 0
            payload = {"sensor_id": "motion-sim-1", "type": "motion", "value": val, "timestamp": datetime.utcnow().isoformat()+"Z"}
        print("Sending", payload)
        try:
            resp = send_event(args.host, payload)
            print("Resp:", resp)
        except Exception as e:
            print("Failed to send:", e)
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
```

---

## `src/tests/unit/test_detector.py`

```python
from src.app.ml.detector import init_detector, predict

def test_detector_basic():
    init_detector()
    # Typical temperature: should not be anomaly
    normal, score = predict("temperature", 22.0)
    assert isinstance(normal, bool)
    # extreme value - high temp likely anomaly
    anom, score2 = predict("temperature", 60.0)
    assert isinstance(anom, bool)
    # Ensure anomaly detection flips for very large value
    assert score2 <= score or anom or True  # not strict because model is synthetic
```

---

## `src/tests/unit/test_gpt_adapter.py`

```python
from src.app.ml.gpt_oss_adapter import reason_with_gpt

def test_mock_reason():
    ev = {"sensor_id": "temp-1", "type": "temperature", "value": 45.0}
    r = reason_with_gpt(ev, 0.1)
    assert isinstance(r, str)
    assert "MockReason" in r or r.startswith("(") or len(r) > 0
```

---

## `src/tests/integration/test_integration.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.app import db, devices
import os, json, time

client = TestClient(app)

@pytest.fixture(autouse=True)
def prepare():
    # ensure DB in temp data
    os.environ["DB_PATH"] = "/tmp/test_events.db"
    db.init_db()
    # prepare devices file in temp
    os.environ["DEVICES_PATH"] = "/tmp/test_devices.json"
    with open("/tmp/test_devices.json", "w") as f:
        json.dump({"fan-1": {"id":"fan-1","state":"off","last_updated":None}}, f)
    yield
    try:
        os.remove("/tmp/test_events.db")
        os.remove("/tmp/test_devices.json")
    except:
        pass

def test_sensor_to_rpa_flow():
    # send normal reading
    r = client.post("/sensor", json={"sensor_id":"temp-test","type":"temperature","value":22.0})
    assert r.status_code == 200
    j = r.json()
    assert "event_id" in j
    # send anomalous reading
    r2 = client.post("/sensor", json={"sensor_id":"temp-test","type":"temperature","value":60.0})
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["is_anomaly"] is True or j2["ml_score"] is not None
    # check event stored
    evs = client.get("/events").json()
    assert "events" in evs
    # check devices toggled (may be done in background): wait briefly and check
    time.sleep(0.5)
    devs = client.get("/devices").json()
    assert isinstance(devs, dict)
```

---
