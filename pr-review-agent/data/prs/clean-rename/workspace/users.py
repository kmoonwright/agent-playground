"""User lookup. `get_user` was renamed to `fetch_user`; call sites updated."""

DB = {
    1: {"name": "Ada", "email": "ada@example.com"},
    2: {"name": "Bob", "email": "bob@example.com"},
}


def fetch_user(user_id: int) -> dict | None:
    """Return the user record, or None if missing."""
    return DB.get(user_id)
