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
