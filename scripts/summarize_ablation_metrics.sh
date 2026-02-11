#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
RESULTS_SUBDIR="${1:-pilot_rule_variant_sweeps_after_fix}"
SUMMARY_DIR="${2:-${OUTPUT_ROOT}/${RESULTS_SUBDIR}/summary}"
RESULTS_DIR="${OUTPUT_ROOT}/${RESULTS_SUBDIR}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Results directory not found: ${RESULTS_DIR}"
  exit 1
fi

mkdir -p "${SUMMARY_DIR}"
RUN_CSV="${SUMMARY_DIR}/run_metrics.csv"
GROUP_CSV="${SUMMARY_DIR}/group_metrics.csv"

echo "group,run,accuracy_test,nll_test,accuracy_valid,nll_valid" > "${RUN_CSV}"

while IFS= read -r final_json; do
  run_dir="$(dirname "$(dirname "${final_json}")")"
  run_name="$(basename "${run_dir}")"

  run_stem="$(echo "${run_name}" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')"
  group="$(echo "${run_stem}" | sed -E 's/_seed[0-9]+$//')"

  acc_test="$(jq -r '.accuracy.test' "${final_json}")"
  nll_test="$(jq -r '.categorical_loglikelihood.test' "${final_json}")"
  acc_valid="$(jq -r '.accuracy.valid' "${final_json}")"
  nll_valid="$(jq -r '.categorical_loglikelihood.valid' "${final_json}")"

  echo "${group},${run_name},${acc_test},${nll_test},${acc_valid},${nll_valid}" >> "${RUN_CSV}"
done < <(find "${RESULTS_DIR}" -type f -path '*/performance/final.json' | sort)

awk -F',' '
NR == 1 { next }
{
  group = $1
  acc = $3 + 0
  nll = $4 + 0
  n[group] += 1
  acc_sum[group] += acc
  acc_sq[group] += acc * acc
  nll_sum[group] += nll
  nll_sq[group] += nll * nll
}
END {
  print "group,n,accuracy_test_mean,accuracy_test_std,nll_test_mean,nll_test_std"
  for (group in n) {
    acc_mean = acc_sum[group] / n[group]
    nll_mean = nll_sum[group] / n[group]
    acc_var = (acc_sq[group] / n[group]) - (acc_mean * acc_mean)
    nll_var = (nll_sq[group] / n[group]) - (nll_mean * nll_mean)
    if (acc_var < 0) acc_var = 0
    if (nll_var < 0) nll_var = 0
    printf "%s,%d,%.6f,%.6f,%.6f,%.6f\n", group, n[group], acc_mean, sqrt(acc_var), nll_mean, sqrt(nll_var)
  }
}
' "${RUN_CSV}" | sort > "${GROUP_CSV}"

echo "Saved run-level metrics: ${RUN_CSV}"
echo "Saved group-level metrics: ${GROUP_CSV}"
