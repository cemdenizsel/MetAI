from fastapi import Request

def is_localhost(request: Request) -> bool:
    hostname = (request.url.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0"}