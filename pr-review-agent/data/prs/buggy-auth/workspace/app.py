"""Tiny HTTP-ish wrapper around login. Rate limiting lives in auth.login."""

from auth import login


def handle_login(body: dict) -> dict:
    username = body.get("username", "")
    password = body.get("password", "")
    return login(username, password)
