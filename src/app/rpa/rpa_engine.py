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
