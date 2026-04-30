# Data Retention Policy

The platform should retain only the data required for analytics, auditing, reliability, and customer support.

Default retention targets:

- API access logs: 30 days
- Audit logs: 1 year
- Aggregated usage metrics: 2 years
- Raw training datasets: project-specific approval required
- Model checkpoints: retain promoted versions and recent candidates only

Sensitive data should be minimized before storage, and retention jobs should be reviewed before production deployment.
