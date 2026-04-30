# Audit Controls

This folder documents how the platform records and reviews security-relevant events.

Audit events should capture:

- actor identifier
- organization identifier
- action performed
- resource type and resource id
- request id
- timestamp
- structured metadata without secrets

Audit logs should be append-only. Application code should not update or delete audit rows after insertion.
