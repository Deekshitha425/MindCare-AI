"""
utils/auth_manager.py
Simple username + hashed-password authentication.
Users are stored in:  project/outputs/users/accounts.json
"""
import os
import json
import hashlib
import streamlit as st

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACCOUNTS_PATH = os.path.join(ROOT, "project", "outputs", "users", "accounts.json")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    """SHA-256 hash of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _load_accounts() -> dict:
    if not os.path.exists(ACCOUNTS_PATH):
        return {}
    with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def _save_accounts(accounts: dict) -> None:
    os.makedirs(os.path.dirname(ACCOUNTS_PATH), exist_ok=True)
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def signup(username: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success: bool, message: str).
    """
    username = username.strip().lower().replace(" ", "_")
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    accounts = _load_accounts()
    if username in accounts:
        return False, f"Username '{username}' is already taken. Please choose another."

    accounts[username] = {"password_hash": _hash(password)}
    _save_accounts(accounts)
    return True, f"Account created for '{username}'! You can now log in."


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Verify credentials.
    Returns (success: bool, message: str).
    """
    username = username.strip().lower().replace(" ", "_")
    if not username or not password:
        return False, "Please enter both username and password."

    accounts = _load_accounts()
    if username not in accounts:
        return False, "Username not found. Please sign up first."
    if accounts[username]["password_hash"] != _hash(password):
        return False, "Incorrect password. Please try again."

    return True, "Login successful."


def is_logged_in() -> bool:
    return bool(st.session_state.get("username"))


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
