#!/bin/bash
# Simple reimbursement calculation
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

days="$1"
miles="$2"
receipts="$3"

# Basic approximation formula
reimbursement=$(echo "scale=2; 100 + 50*$days + 0.6*$miles + $receipts" | bc)

echo "$reimbursement"
