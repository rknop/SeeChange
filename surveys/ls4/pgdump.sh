#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: pgdump.sh <password> <outfile>"
    exit 1
fi

PGPASSWORD=$1 pg_dump -h ls4db.lbl.gov -U ls4_production --format=c --file=$2
