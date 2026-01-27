#!/bin/bash
set -e

# This script runs on every container start to ensure pg_hba.conf allows non-SSL connections
if [ -f "$PGDATA/pg_hba.conf" ]; then
    # Check if the entry already exists
    if ! grep -q "172.16.0.0/12" "$PGDATA/pg_hba.conf"; then
        echo "host    all             all             172.16.0.0/12          trust" >> "$PGDATA/pg_hba.conf"
        echo "host    all             all             10.0.0.0/8             trust" >> "$PGDATA/pg_hba.conf"
        echo "host    all             all             172.18.0.0/16          trust" >> "$PGDATA/pg_hba.conf"
        # Reload PostgreSQL configuration
        psql -U postgres -c "SELECT pg_reload_conf();" || true
    fi
fi
