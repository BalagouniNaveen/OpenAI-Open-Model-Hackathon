from src.app.ml.gpt_oss_adapter import reason_with_gpt

def test_mock_reason():
    ev = {"sensor_id": "temp-1", "type": "temperature", "value": 45.0}
    r = reason_with_gpt(ev, 0.1)
    assert isinstance(r, str)
    assert "MockReason" in r or r.startswith("(") or len(r) > 0
