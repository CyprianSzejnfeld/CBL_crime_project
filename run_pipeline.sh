#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
PY=backend/.venv/bin/python

run() {
  echo "=== $1 ($(date +%H:%M:%S)) ==="
  $PY -m "$1"
  rc=$?
  echo "--- exit $rc"
  if [ $rc -ne 0 ]; then exit $rc; fi
}

run src.build_london_panel
run src.search_metrics
run src.build_stop_search_categories
run src.process_mopac_trust_context

run src.data_interfaces
run src.crime_forecast
run src.crime_model_v2
run src.fairness_v2.fairness_engine
run src.intervention_geography
run src.clustering.build
run src.search_regimes
run src.protection_need
run src.packages.eligibility
run src.optimisation.run
run src.app_data
echo "PIPELINE COMPLETE"
