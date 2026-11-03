"""Bearer token handling. Tokens are HS256 JWTs with a `role` claim."""

import jwt

ALGORITHM = "HS256"


class InvalidToken(Exception):
    pass


def role_from_header(header: str | None, secret: str) -> str:
    """Returns the role in the token, or raises InvalidToken."""
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise InvalidToken("missing bearer token")
    try:
        # Pinning the algorithm blocks alg=none and key-confusion tokens.
        claims = jwt.decode(token.strip(), secret, algorithms=[ALGORITHM], options={"require": ["exp"]})
    except jwt.InvalidTokenError as exc:
        raise InvalidToken(str(exc)) from exc

    role = claims.get("role")
    if not isinstance(role, str) or not role:
        raise InvalidToken("token has no role claim")
    return role


def issue_token(role: str, secret: str, expires_at: int) -> str:
    """For local testing. A real deployment gets tokens from its identity provider."""
    return jwt.encode({"role": role, "exp": expires_at}, secret, algorithm=ALGORITHM)
