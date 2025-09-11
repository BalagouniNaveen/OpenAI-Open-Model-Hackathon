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




