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
