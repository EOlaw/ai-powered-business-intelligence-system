# Database Migrations

Timestamped migration folders belong here.

Create a new migration stub with:

```bash
python scripts/migrations/create_migration.py "add usage rollups"
```

Each migration folder should include a `migration.sql` file and a short note in the pull request explaining the operational impact.
