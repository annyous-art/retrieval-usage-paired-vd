#!/usr/bin/env bash
# Rebuild every table, figure, and number of the paper from the shipped answers (offline, no API calls).
# Needs PrimeVul's paired test file and MegaVul's Java release (see README):
#   bash reproduce.sh path/to/primevul_test_paired.jsonl path/to/megavul.json
set -euo pipefail
cd "$(dirname "$0")"
A=analysis/fse_revision_20260923
python3 tools/build_labeled_test.py "${1:-data/primevul_test_paired.jsonl}"
python3 tools/build_megavul_java.py "${2:-data/second_dataset/megavul_java/megavul.json}"
python3 tools/expand_predictions.py > /dev/null
for s in literature_protocol cwe_label_agreement \
         rq2_missing_rows e_rq2_retrieval rq2_similarity_scores focus_localization rq2_match_by_member java_rq1_retrieval \
         rq1_rq3_modes gpt_std_prompts glm_std_prompts vulrag_comparison vulrag_evidence_identity \
         rq2_zero_shot_subsets rq2_holm oracle_arms java_figure_data java_subsets \
         rq3_figure_data rq3_holm bias_controls nondeterminism_floor glm_repeat; do
  echo "== $s"
  python3 $A/$s.py > /dev/null
done
python3 $A/make_rq_tables.py tables
python3 figures/plot_rq2.py
python3 figures/plot_java.py
python3 figures/plot_rq3.py
