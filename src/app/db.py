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
