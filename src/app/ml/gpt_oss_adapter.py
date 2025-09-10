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
