#!/usr/bin/env bash
# Local CPU pipeline: gate -> parse -> controls -> metrics -> report.
# The GPU stages are per-model and live in the notebooks.
#
#   ./scripts/run_pipeline.sh /path/to/FaceForensics++_C23 ./audit_work [MAX_PAIRS]
set -euo pipefail

DATA="${1:?usage: run_pipeline.sh <dataset root> <work dir> [max_pairs]}"
WORK="${2:?usage: run_pipeline.sh <dataset root> <work dir> [max_pairs]}"
MAX_PAIRS="${3:-0}"
PY="${PYTHON:-python}"
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK"

echo "=== self test ==="
"$PY" "$REPO_ROOT/scripts/selftest.py"

echo "=== module 1: pairing gate ==="
"$PY" -m ccaudit.m1_verify --root "$DATA" --sample 40 --probes 3 \
    --json "$WORK/m1_report.json" --emit-pairs "$WORK/pairs.json" --fail-hard

echo "=== module 2: parse ==="
"$PY" -m ccaudit.m2_parse --pairs "$WORK/pairs.json" --out "$WORK/parsed" \
    --data-root "$DATA" --max-pairs "$MAX_PAIRS" --workers "$NPROC" \
    --overlay "$WORK/overlay.png"

echo "=== module 5: control detectors ==="
for i in $(seq 0 $((NPROC-1))); do
  "$PY" -m ccaudit.m5_runner --index "$WORK/parsed/index.json" \
      --detector "adaptive_oracle,fixed_oracle:mouth,confabulator,dummy" \
      --out "$WORK/run_controls" --tag main --splice-floor --inpaint telea \
      --shard "$i/$NPROC" --device cpu > "$WORK/controls_$i.log" 2>&1 &
done
wait

echo "=== module 10: localization ==="
"$PY" -m ccaudit.m10_localization --index "$WORK/parsed/index.json" --out "$WORK/loc"

echo "=== module 6: metrics ==="
"$PY" -m ccaudit.m6_metrics --raw "$WORK/run_controls" --out "$WORK/metrics" \
    --localization "$WORK/loc/localization.json" --by method
"$PY" -m ccaudit.m6_metrics --raw "$WORK/run_controls" --out "$WORK/metrics" --coarse

echo "=== module 7: report ==="
"$PY" -m ccaudit.m7_report --metrics "$WORK/metrics/metrics.json" \
    --coarse "$WORK/metrics/metrics_coarse.json" \
    --by-method "$WORK/metrics/metrics_by_method.json" \
    --raw "$WORK/run_controls" --out "$WORK/report.html" \
    --figures-dir "$WORK/figures"

echo
echo "done -> $WORK/report.html"
echo "Section 2 of the report shows the control detectors; their ordering must hold."
