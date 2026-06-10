#!/usr/bin/env bash
# Run the remaining long rework phases unattended.
#
# This wrapper intentionally calls run_rework.sh phase-by-phase instead of
# duplicating phase logic here. Each phase writes its own ignored run_*.log file.

set -euo pipefail

CARBON_PRICE="${CARBON_PRICE:-80}"
WORKERS="${WORKERS:-10}"
VAL_FRAC="${VAL_FRAC:-0.05}"
PRICES_AXIS="${PRICES_AXIS:-0 20 40 60 80 100 120 160}"
FAB_WINDOWS="${FAB_WINDOWS:-12}"
PHASES="${PHASES:-cstr ensemble manifest}"

export CARBON_PRICE WORKERS VAL_FRAC PRICES_AXIS FAB_WINDOWS

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"

stamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

run_phase() {
  local phase="$1"
  local log="$LOG_DIR/run_overnight_${phase}.log"

  echo
  echo "[$(stamp)] starting phase: $phase"
  echo "[$(stamp)] log: $log"

  ./run_rework.sh "$phase" 2>&1 | tee "$log"

  echo "[$(stamp)] finished phase: $phase"
}

echo "=== run_rework_overnight.sh ==="
echo "started=$(stamp)"
echo "phases=$PHASES"
echo "carbon_price=$CARBON_PRICE workers=$WORKERS val_frac=$VAL_FRAC"
echo "prices_axis=$PRICES_AXIS"
echo "fab_windows=$FAB_WINDOWS"

if command -v caffeinate >/dev/null 2>&1; then
  echo "caffeinate=available"
  caffeinate -dimsu -w "$$" &
  CAFFEINATE_PID="$!"
  trap 'kill "$CAFFEINATE_PID" >/dev/null 2>&1 || true' EXIT
else
  echo "caffeinate=not available"
fi

for phase in $PHASES; do
  run_phase "$phase"
done

echo
echo "completed=$(stamp)"
echo "All requested overnight phases completed."
