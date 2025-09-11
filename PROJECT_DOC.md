# **IoT + ML + RPA Project Documentation**

## **1. Project Overview**

This project is a **simulated IoT system** integrated with:

* **ML Anomaly Detection** (IsolationForest)
* **RPA Automation** (device toggles, safe scripts)
* **GPT-OSS Reasoning** for human-readable anomaly explanations

**Architecture Flow:**

```
Sensors → API → DB → ML Detector → GPT-OSS Reasoner → RPA Engine → Devices / Logs / Dashboard
```

* Sensors send readings via `POST /sensor`.
* API stores events and triggers ML detection.
* ML Detector flags anomalies and scores readings.
* GPT-OSS Adapter explains anomalies (mocked by default).
* RPA Engine performs safe actions (device toggles or scripts).
* Dashboard displays live events and device states.

---

## **2. Directory Structure & Purpose**

```
data/
  devices.json           # Device states
  sample_sensor_data.json

docs/
  architecture.md        # System overview

scripts/
  safe_action.sh         # Safe demo script
  run_local_model.sh     # GPT-OSS setup script

src/app/
  main.py                # FastAPI server entry
  api.py                 # API endpoints
  db.py                  # SQLite DB functions
  devices.py             # Device state management
  ml/
    detector.py          # Anomaly detection
    trainer.py           # Synthetic model generation
    gpt_oss_adapter.py   # Reasoning (mock or GPT-OSS)
  rpa/
    rpa_engine.py        # Executes RPA actions
  dashboard/
    templates/index.html # Dashboard page
    static/app.js        # Frontend logic

src/sensors/
  simulator.py           # Sensor event simulation

src/tests/
  test_detector.py       # ML unit tests
  test_gpt_adapter.py    # GPT reasoning unit tests
  test_integration.py    # End-to-end workflow tests
```

---

## **3. Step-by-Step Execution**

### **Step 1: Setup Environment**

1. Install Python 3.11.
2. Create a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
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

1. Initialize SQLite DB:

```bash
python src/app/db.py
```

2. Train / generate synthetic IsolationForest model:

```bash
python src/app/ml/trainer.py
```

---

### **Step 3: Start FastAPI Server**

```bash
uvicorn src.app.main:app --reload
```

* Dashboard: [http://localhost:8000](http://localhost:8000)
* API Endpoints:

  * `POST /sensor` → submit events
  * `GET /events` → list events
  * `GET /devices` → device states

---

### **Step 4: Simulate Sensor Data**

```bash
python src/sensors/simulator.py --count 5 --type temperature
python src/sensors/simulator.py --count 3 --type motion
```

* Events sent to API via `POST /sensor`
* Stored in SQLite and processed by ML detector

---

### **Step 5: ML Detection & GPT-OSS Reasoning**

* ML Detector flags anomalies:

```text
is_anomaly → True/False
ml_score → Numeric anomaly score
```

* GPT-OSS Adapter provides explanation:

```text
"MockReason: Possible overheating. Turn on fan. (score=-0.7)"
```

---

### **Step 6: RPA Action Trigger**

* Temperature anomaly → toggle devices (`fan-1`, `heater-1`)
* Motion anomaly → run safe scripts in `scripts/`
* Logs actions in `rpa.log`

---

### **Step 7: View Dashboard & Events**

* Dashboard shows:

  * Devices states (`on/off`) with last updated time
  * Last 100 sensor events with `is_anomaly`, `ml_score`, `gpt_reason`, and RPA actions

* API example:

```bash
curl http://localhost:8000/events
curl http://localhost:8000/devices
```

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

* Expected:

  * ML correctly flags anomalies
  * GPT reasoning returns explanation
  * RPA actions performed safely
  * Dashboard updated

---

## **4. Expected Outputs (Tested Flow)**

1. **Normal Temperature Event (22°C)**

```json
{
  "event_id": 1,
  "is_anomaly": false,
  "ml_score": 0.25,
  "gpt_reason": "MockReason: Value within expected range. (score=0.25)",
  "rpa": null
}
```

2. **High Temperature Anomaly (45°C)**

```json
{
  "event_id": 2,
  "is_anomaly": true,
  "ml_score": -0.7,
  "gpt_reason": "MockReason: Possible overheating. Turn on fan / shut down non-critical load. (score=-0.7)",
  "rpa": {
    "scheduled": true,
    "action": {"type": "toggle_device", "device_id": "fan-1", "state": "on"}
  }
}
```

3. **Motion Event (unexpected)**

```json
{
  "event_id": 3,
  "is_anomaly": true,
  "ml_score": -0.9,
  "gpt_reason": "MockReason: Motion event: verify scheduled occupancy; if unexpected, send alert. (score=-0.9)",
  "rpa": {
    "scheduled": true,
    "action": {"type": "run_script", "script": "safe_action.sh", "args": ["motion-sim-1"]}
  }
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

## **5. Key Takeaways**

* End-to-end functional: **IoT → ML → GPT reasoning → RPA → Dashboard**
* Safe for demo: uses mock GPT reasoning and simulated devices
* Modular: easy to extend with real IoT, GPT-OSS, or multi-step automation
* Hackathon-ready with **dashboard visualization and RPA demo**

---

Do you want me to prepare that diagram next?
