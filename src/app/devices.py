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
