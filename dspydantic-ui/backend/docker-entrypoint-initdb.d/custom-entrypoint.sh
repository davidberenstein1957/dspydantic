#!/bin/bash
set -e

# Run the original postgres entrypoint
/docker-entrypoint.sh "$@" &

# Wait for postgres to start
until pg_isready -U postgres; do
  sleep 1
done

# Fix pg_hba.conf to allow non-SSL connections from Docker networks
if [ -f "$PGDATA/pg_hba.conf" ]; then
    # Remove any existing entries for our networks and add new ones
    sed -i '/172\.16\.0\.0\/12/d' "$PGDATA/pg_hba.conf"
    sed -i '/172\.18\.0\.0\/16/d' "$PGDATA/pg_hba.conf"
    sed -i '/10\.0\.0\.0\/8/d' "$PGDATA/pg_hba.conf"
    
    # Add entries for Docker networks (non-SSL)
    echo "host    all             all             172.16.0.0/12          trust" >> "$PGDATA/pg_hba.conf"
    echo "host    all             all             172.18.0.0/16          trust" >> "$PGDATA/pg_hba.conf"
    echo "host    all             all             10.0.0.0/8             trust" >> "$PGDATA/pg_hba.conf"
    
    # Reload configuration
    psql -U postgres -c "SELECT pg_reload_conf();" || true
fi

wait
