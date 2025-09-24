#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <country> <current_date>"
    exit 1
fi

# Assign input parameters to variables
country=$1
current_date=$2

# Define the final end date
final_end_current_date="2025-01-19"

# Loop until end_current_date reaches final_end_current_date
while [[ "$current_date" < "$final_end_current_date" ]]; do
    echo "Running command with country=$country, current_date=$current_date"
    python -m projects.vendor_ranking.two_towers_v1.model_monitoring.main "$country" \
        --current_date "$current_date"

    current_date=$(date -j -v+1d -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")
done
