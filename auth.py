# auth.py

import json
import os
import hashlib
import secrets
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_db() -> dict:
    """Load the JSON database; seed two default users if the file doesn't exist."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── seed data ──────────────────────────────────────────────
    db = {
        "users": {
            "alice": {
                "password": _hash_password("alice123"),
                "nickname": "Alice",
                "created_at": datetime.now().isoformat(),
                "friends": ["bob"],
                "friend_requests_in": [],
                "friend_requests_out": [],
            },
            "bob": {
                "password": _hash_password("bob123"),
                "nickname": "Bob",
                "created_at": datetime.now().isoformat(),
                "friends": ["alice"],
                "friend_requests_in": [],
                "friend_requests_out": [],
            },
        },
        "tokens": {},  # token -> username
    }
    _save_db(db)
    return db


def _save_db(db: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════

class Auth:

    @staticmethod
    def login(username: str, password: str) -> dict:
        db = _load_db()
        user = db["users"].get(username)
        if not user or user["password"] != _hash_password(password):
            return {"success": False, "error": "Invalid username or password"}

        token = secrets.token_hex(16)
        db["tokens"][token] = username
        _save_db(db)
        return {
            "success": True,
            "token": token,
            "username": username,
            "nickname": user["nickname"],
        }

    @staticmethod
    def register(username: str, password: str, nickname: str = "") -> dict:
        db = _load_db()
        if username in db["users"]:
            return {"success": False, "error": "Username already exists"}
        if len(username) < 2 or len(password) < 4:
            return {"success": False, "error": "Username must be at least 2 characters, password at least 4"}

        db["users"][username] = {
            "password": _hash_password(password),
            "nickname": nickname or username,
            "created_at": datetime.now().isoformat(),
            "friends": [],
            "friend_requests_in": [],
            "friend_requests_out": [],
        }
        _save_db(db)

        token = secrets.token_hex(16)
        db["tokens"][token] = username
        _save_db(db)
        return {
            "success": True,
            "token": token,
            "username": username,
            "nickname": nickname or username,
        }

    @staticmethod
    def verify_token(token: str) -> str | None:
        """Return username if valid, else None."""
        db = _load_db()
        return db["tokens"].get(token)

    @staticmethod
    def logout(token: str):
        db = _load_db()
        db["tokens"].pop(token, None)
        _save_db(db)


# ══════════════════════════════════════════════════════════════
# FRIENDS
# ══════════════════════════════════════════════════════════════

class FriendManager:

    @staticmethod
    def send_request(from_user: str, to_user: str) -> dict:
        db = _load_db()
        if to_user not in db["users"]:
            return {"success": False, "error": "User not found"}
        if from_user == to_user:
            return {"success": False, "error": "You cannot add yourself"}

        me     = db["users"][from_user]
        target = db["users"][to_user]

        if to_user in me["friends"]:
            return {"success": False, "error": "Already friends"}
        if to_user in me["friend_requests_out"]:
            return {"success": False, "error": "Friend request already sent"}

        # if target already sent me a request → auto-accept
        if from_user in target["friend_requests_out"]:
            me["friends"].append(to_user)
            target["friends"].append(from_user)
            target["friend_requests_out"].remove(from_user)
            me["friend_requests_in"] = [
                u for u in me["friend_requests_in"] if u != to_user
            ]
            _save_db(db)
            return {"success": True, "message": f"You are now friends!"}

        me["friend_requests_out"].append(to_user)
        target["friend_requests_in"].append(from_user)
        _save_db(db)
        return {"success": True, "message": f"Friend request sent to {to_user}"}

    @staticmethod
    def accept_request(username: str, from_user: str) -> dict:
        db = _load_db()
        me     = db["users"][username]
        sender = db["users"].get(from_user)

        if not sender:
            return {"success": False, "error": "User not found"}
        if from_user not in me["friend_requests_in"]:
            return {"success": False, "error": "No friend request from this user"}

        me["friend_requests_in"].remove(from_user)
        sender["friend_requests_out"].remove(username)
        me["friends"].append(from_user)
        sender["friends"].append(username)
        _save_db(db)
        return {"success": True, "message": f"Added {from_user} as a friend"}

    @staticmethod
    def reject_request(username: str, from_user: str) -> dict:
        db = _load_db()
        me     = db["users"][username]
        sender = db["users"].get(from_user)

        if not sender:
            return {"success": False, "error": "User not found"}
        if from_user not in me["friend_requests_in"]:
            return {"success": False, "error": "No friend request from this user"}

        me["friend_requests_in"].remove(from_user)
        sender["friend_requests_out"].remove(username)
        _save_db(db)
        return {"success": True, "message": f"Rejected friend request from {from_user}"}

    @staticmethod
    def remove_friend(username: str, friend: str) -> dict:
        db = _load_db()
        me   = db["users"][username]
        them = db["users"].get(friend)

        if not them or friend not in me["friends"]:
            return {"success": False, "error": "This user is not your friend"}

        me["friends"].remove(friend)
        them["friends"].remove(username)
        _save_db(db)
        return {"success": True, "message": f"Removed {friend} from friends"}

    @staticmethod
    def get_friends(username: str) -> dict:
        db   = _load_db()
        user = db["users"][username]
        return {
            "friends": user["friends"],
            "requests_in": user["friend_requests_in"],
            "requests_out": user["friend_requests_out"],
        }
