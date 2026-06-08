"""
utils/history_manager.py
Load and save conversation / assessment history — per-user JSON files.

File layout:
  project/outputs/users/<username>/assessment_history.json
  project/outputs/users/<username>/conversation_history.json
"""
import os
import json
import datetime
import streamlit as st
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USERS_DIR = os.path.join(ROOT, "project", "outputs", "users")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_dir(username: str) -> str:
    path = os.path.join(USERS_DIR, username)
    os.makedirs(path, exist_ok=True)
    return path

def _conv_path(username: str) -> str:
    return os.path.join(_user_dir(username), "conversation_history.json")

def _assessment_path(username: str) -> str:
    return os.path.join(_user_dir(username), "assessment_history.json")

def _current_user() -> str:
    return st.session_state.get("username", "guest")

def _load(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def _save(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Conversation history ──────────────────────────────────────────────────────

def load_conversation_history() -> List[Dict]:
    return _load(_conv_path(_current_user()))

def save_conversation_turn(turn: Dict) -> None:
    path = _conv_path(_current_user())
    history = _load(path)
    history.append(turn)
    _save(path, history)

def clear_conversation_history() -> None:
    _save(_conv_path(_current_user()), [])


# ── Assessment history ────────────────────────────────────────────────────────

def load_assessment_history() -> List[Dict]:
    return _load(_assessment_path(_current_user()))

def save_assessment_record(record: Dict) -> None:
    path = _assessment_path(_current_user())
    history = _load(path)
    history.append(record)
    _save(path, history)
