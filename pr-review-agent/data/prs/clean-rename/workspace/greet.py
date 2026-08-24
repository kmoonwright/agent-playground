from users import fetch_user


def greet(user_id: int) -> str:
    user = fetch_user(user_id)
    if user is None:
        return "hello, stranger"
    return f"hello, {user['name']}"
