"""
InsightSerenity AI Engine — Internal API Key Middleware
=======================================================
The AI engine is NOT a public-facing API. It sits behind the Node.js
API Gateway which handles:
    - Public INSIGHTSERENITY_API_KEY validation
    - Rate limiting, usage tracking, billing checks
    - Routing to the correct AI engine endpoint

The AI engine only needs to verify that the request came from our
own trusted API Gateway — not from an arbitrary external client.

Authentication model (two layers):
    Layer 1 (public):  Node.js gateway validates INSIGHTSERENITY_API_KEY
    Layer 2 (internal): AI engine validates INTERNAL_SERVICE_SECRET
                        This is a shared secret between the gateway and engine.

The internal secret is:
    - Set via SERVING_INTERNAL_API_SECRET environment variable
    - Passed as "Authorization: Bearer <secret>" by the Node.js gateway
    - Never exposed to external clients

Additional context the gateway passes:
    X-Org-Id:    Organization ID (for usage attribution)
    X-Key-Id:    API key ID that the user authenticated with
    X-Scopes:    Comma-separated API key scopes

FastAPI dependency injection:
    async def my_endpoint(auth: AuthContext = Depends(require_internal_auth)):
        ...
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuthContext:
    """
    Authentication context passed to route handlers.

    Populated by the require_internal_auth dependency after validating
    the internal service token and parsing gateway-provided headers.
    """
    org_id:   Optional[str] = None
    key_id:   Optional[str] = None
    scopes:   list           = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []

    def has_scope(self, scope: str) -> bool:
        """Check if a specific scope is granted."""
        return scope in self.scopes or "admin:all" in self.scopes


async def require_internal_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_org_id: Optional[str]     = Header(default=None, alias="X-Org-Id"),
    x_key_id: Optional[str]     = Header(default=None, alias="X-Key-Id"),
    x_scopes: Optional[str]     = Header(default=None, alias="X-Scopes"),
) -> AuthContext:
    """
    FastAPI dependency: verify the internal service token.

    The Node.js gateway sends:
        Authorization: Bearer <SERVING_INTERNAL_API_SECRET>
        X-Org-Id: <org_id>
        X-Key-Id: <api_key_id>
        X-Scopes: <scope1,scope2,...>

    This dependency:
        1. Extracts the Bearer token from Authorization header
        2. Compares it against the configured internal secret
        3. Returns AuthContext with org/key/scope metadata

    Raises:
        HTTPException(401): If the token is missing or invalid.
    """
    # Skip auth in development mode when internal secret is the default
    if settings.is_development and settings.serving.internal_api_secret == "change-me-in-production":
        logger.debug("Auth skipped in development mode")
        return AuthContext(
            org_id=x_org_id or "dev-org",
            key_id=x_key_id or "dev-key",
            scopes=x_scopes.split(",") if x_scopes else ["*"],
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected: Bearer <token>",
        )

    token = parts[1].strip()

    # Constant-time comparison to prevent timing attacks
    import hmac
    expected = settings.serving.internal_api_secret
    if not hmac.compare_digest(token.encode(), expected.encode()):
        logger.warning(
            "Internal auth failed",
            client_ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal service token",
        )

    scopes = x_scopes.split(",") if x_scopes else []
    return AuthContext(org_id=x_org_id, key_id=x_key_id, scopes=scopes)


# Convenience alias for use in route dependencies
InternalAuth = Depends(require_internal_auth)
