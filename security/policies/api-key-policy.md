# API Key Policy

API keys authenticate external clients and must be treated as production secrets.

Requirements:

- Store only hashed API keys.
- Show plaintext keys once at creation.
- Support revocation and rotation.
- Scope keys by capability.
- Rate limit keys by organization and plan.
- Record last-used timestamps for operational review.

Keys should never be committed to Git, printed in logs, or returned by list endpoints.
