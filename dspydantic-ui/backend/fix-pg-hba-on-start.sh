#!/bin/bash
set -e

# Wait for PostgreSQL to be ready (with timeout)
for i in {1..30}; do
    if pg_isready -U postgres > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Wait a bit more for database initialization
sleep 2

# Fix pg_hba.conf to allow non-SSL connections from Docker networks
PGDATA=${PGDATA:-/var/lib/postgresql/data}

if [ -f "$PGDATA/pg_hba.conf" ]; then
    # Remove any existing entries for Docker networks
    sed -i '/172\.16\.0\.0\/12/d' "$PGDATA/pg_hba.conf" || true
    sed -i '/172\.18\.0\.0\/16/d' "$PGDATA/pg_hba.conf" || true
    sed -i '/10\.0\.0\.0\/8/d' "$PGDATA/pg_hba.conf" || true
    
    # Add entries for Docker networks (non-SSL, trust authentication)
    echo "host    all             all             172.16.0.0/12          trust" >> "$PGDATA/pg_hba.conf"
    echo "host    all             all             172.18.0.0/16          trust" >> "$PGDATA/pg_hba.conf"
    echo "host    all             all             10.0.0.0/8             trust" >> "$PGDATA/pg_hba.conf"
    
    # Reload configuration (use template1 which always exists)
    psql -U postgres -d template1 -c "SELECT pg_reload_conf();" 2>/dev/null || true
fi
