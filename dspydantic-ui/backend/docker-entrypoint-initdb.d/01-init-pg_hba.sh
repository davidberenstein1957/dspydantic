#!/bin/bash
set -e

# Configure pg_hba.conf to allow non-SSL connections from Docker networks
if [ -f "$PGDATA/pg_hba.conf" ]; then
    # Backup original
    cp "$PGDATA/pg_hba.conf" "$PGDATA/pg_hba.conf.backup"
    
    # Add entry for Docker networks (non-SSL)
    echo "host    all             all             172.16.0.0/12          trust" >> "$PGDATA/pg_hba.conf"
    echo "host    all             all             10.0.0.0/8             trust" >> "$PGDATA/pg_hba.conf"
fi
