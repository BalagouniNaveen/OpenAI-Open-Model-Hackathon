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
