"""
gateway/auth.py
───────────────
DEV MODE — Dummy authentication with 3 hardcoded users.

Pass one of these tokens in the Authorization: Bearer header:

    User          Token
    ──────────    ─────────────
    arjun_singh   dev-token-arjun
    priya_nair    dev-token-priya
    vikram_rao    dev-token-vikram

Any other token → HTTP 401.

TODO: Replace _DUMMY_USERS with real JWT decode when going to production.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("gateway.auth")

# ── Hardcoded dev users ───────────────────────────────────────────────────────
_DUMMY_USERS: dict[str, str] = {
    "dev-token-arjun": "arjun_singh",
    "dev-token-priya": "priya_nair",
    "dev-token-vikram": "vikram_rao",
}

_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    FastAPI dependency — call as:
        user_id: str = Depends(get_current_user)

    Returns the username for the matching dummy token.
    Raises HTTP 401 for any unrecognised token.

    Test with curl:
        curl -H "Authorization: Bearer dev-token-arjun" http://localhost:8000/threads
    """
    token = credentials.credentials
    username = _DUMMY_USERS.get(token)

    if not username:
        log.warning(f"Unknown dev token: {token!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Unrecognised token. "
                "Valid dev tokens: dev-token-arjun, dev-token-priya, dev-token-vikram"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    log.info(f"🔓 Auth OK → {username}")
    return username
