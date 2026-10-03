# Does Retrieval Help LLMs Distinguish Vulnerable Code from Its Fix? Replication Package

Anonymous replication package for the paper *Does Retrieval Help LLMs Distinguish Vulnerable Code
from Its Fix? An Empirical Study on Paired Benchmarks*. It contains the coded literature list, the
retrieval corpora and retrieval outputs, the code that builds the prompts and calls the models,
every model answer used in the paper (without the code of the benchmarks), and the analysis and
plotting scripts that produce the paper's tables, figures, and numbers. No API credentials or
service endpoints are included.

## Quick check: reproduce the paper's tables, figures, and numbers

```bash
pip install -r requirements.txt
bash reproduce.sh path/to/primevul_test_paired.jsonl path/to/megavul.json
```

`reproduce.sh` runs offline and makes no API calls. It

1. rebuilds the labeled PrimeVul test file (`tools/build_labeled_test.py`) and the MegaVul Java pairs
   (`tools/build_megavul_java.py`) from the benchmarks' own releases and our line labels, checking the
   SHA-256 of every input and output;
2. restores the run outputs that the analyses read from `predictions/` (`tools/expand_predictions.py`);
3. runs the analysis scripts in `analysis/fse_revision_20260923/`, which rewrite their JSON results;
4. writes the LaTeX tables to `tables/` (`make_rq_tables.py`) and the figures to `figures/`.

With the shipped answers, every JSON result is byte-identical to the shipped one, the four tables are
identical to the paper's LaTeX source, and the three figures are pixel-identical to the paper's
(with Linux Libertine from TeX Live installed; otherwise a serif fallback font is used).

## Paper results and scripts

All scripts are in `analysis/fse_revision_20260923/` unless noted, and run from the repository root.

| Paper | Script(s) |
| --- | --- |
| Section 2.2, Table 1 (literature search, usage modes) | `literature_protocol.py` |
| Section 4.1 (CWE agreement of PrimeVul and NVD, 61%) | `cwe_label_agreement.py` |
| Section 4.1 (MegaVul Java split: 1,564 / 620 pairs, 251 commits, 118 projects) | `tools/build_megavul_java.py` (`data/second_dataset/megavul_java_split_stats.json`) |
| Table 2 (RQ1: same items, close scores, match) | `rq2_missing_rows.py` (A, C/D, Vul-RAG), `java_rq1_retrieval.py` (B same items), `rq2_similarity_scores.py` (B scores), `e_rq2_retrieval.py` (E), `rq2_match_by_member.py` (match); then `make_rq_tables.py` |
| RQ1 text: focus region of B (26.2% vs. 24.8% for a random region) | `focus_localization.py` |
| RQ1 text: MegaVul same items, focus misses (34.7% vs. 65.2%), function lengths | `java_rq1_retrieval.py` |
| Table 3 (pair outcomes, GPT-5.5 and GLM-5.1) | `rq1_rq3_modes.py`, `gpt_std_prompts.py` and `glm_std_prompts.py` (zero-shot), `vulrag_comparison.py` (Vul-RAG); then `make_rq_tables.py` |
| Figure 3a (net P-C vs. zero-shot, same / different items) | `rq2_zero_shot_subsets.py`; Holm correction `rq2_holm.py`; plot `figures/plot_rq2.py` |
| Figure 3b (control experiment with the pair's own fix, and unrelated-pair evidence) | `oracle_arms.py`; plot `figures/plot_rq2.py` |
| RQ2 text: Vul-RAG with GLM-5.1, and with another pair's knowledge | `vulrag_comparison.py` (`glm-5.1`) |
| Table 4 (Claude Opus 4.7, DeepSeek-V4-Pro) | `rq1_rq3_modes.py`; then `make_rq_tables.py` |
| Figure 4 (MegaVul Java pairs) | `java_figure_data.py`; plot `figures/plot_java.py`; B and E within same / different items: `java_subsets.py` |
| Figure 5 (RQ3, presentation of the same items) | `rq3_figure_data.py` (with `rq3_order.py`); Holm correction `rq3_holm.py`; plot `figures/plot_rq3.py`; F1 of the regrouped chunks: `rq1_rq3_modes.py` |
| Table 5 (context controls, a separate batch of runs) | `bias_controls.py`; then `make_rq_tables.py` |
| Section 7 (variation between runs: two GLM-5.1 prompts sent a second time) | `glm_repeat.py` |

Pair-level statistics use the 431 resolvable PrimeVul pairs (four pairs touch the two test indices
that occur twice) and the 620 MegaVul Java pairs, a commit-cluster bootstrap with 3,000 draws (seed
20260920), and Holm correction within the families named in the paper.

The other scripts in `analysis/fse_revision_20260923/` (for example `ablation_cluster_ci.py`,
`evidence_mechanism.py`, `mechanism_cross_model.py`, `presentation_variants.py`, `shared_knowledge.py`,
`nondeterminism_floor.py` (P-C on identical prompts), `per_method_taxonomy/`) are supplementary analyses from earlier drafts; their results are kept as
JSON. Those that inspect the full prompt text (`evidence_mechanism.py`, `mechanism_cross_model.py`)
need the full run outputs.

## Layout

| Path | Content |
| --- | --- |
| `literature/` | Literature search: Semantic Scholar bulk-search results, forward-citation lists, and the coded list of the 30 included studies (`included_studies.csv`: usage modes A–E and whether a study is an empirical evaluation). |
| `data/primevul_test_paired_labels.jsonl` | Our diff-derived line labels for the 870 functions of the PrimeVul paired test split, keyed by `idx` and `func_hash`; merged into PrimeVul's own file by `tools/build_labeled_test.py`. |
| `data/second_dataset/` | MegaVul Java pairs: the build scripts (`build/`), our line labels (`megavul_java_paired_labels_diff.jsonl.gz`, computed as for PrimeVul by `experiments/java_b_relabel/relabel_java_diff.py`), and the split statistics. MegaVul's data is not redistributed. |
| `data/e_retrieval/`, `data/e_retrieval_java/` | Mode E corpora and retrieval results for PrimeVul and MegaVul: MITRE CWE 4.20 entries (756 C/C++ and 748 Java entries), CVE descriptions of the training splits, the NVD CWE mapping of the PrimeVul test CVEs, and the entries retrieved per test function; `manifest.json` lists SHA-256 hashes. |
| `data/b_keys/` | For every test function of both datasets, the SHA-256 of its mode B evidence and the line numbers of its focus region (`tools/export_b_keys.py`); no code. |
| `data/retrieval_scores.json` | Top-1 retrieval scores of modes A and C/D and the CVEs of the retrieved items (written by `rq2_missing_rows.py` from the embeddings and the candidate pool, which contain code). |
| `predictions/` | Every run used in the paper, without the code: compact tables (`.csv.gz`) with function index, label, arm, ids of the retrieved items, raw model answer, parsed YES/NO prediction, and `extra` (job, row and pair ids; SHA-256 of a mode B evidence block; for three mode B runs, the code-free retrieval records); and code-free auxiliary records (`.gz`: jobs, targets, retrieval lists, configs). Paths mirror the original output paths. |
| `outdir/` | Small outputs kept as they are: code-free control runs (Table 5), the Vul-RAG runs (`vulrag_gpt-5.5/`: targets, queries, retrieved knowledge ids, decisions, and the extracted knowledge base; `vulrag_glm51_*`: GLM-5.1 with GPT-5.5's knowledge base, with the retrieved knowledge and with another pair's), the metrics of the control experiment with the pair's own fix (`oracle_fc_*`), and the error-taxonomy sample (earlier draft). |
| `analysis/fse_revision_20260923/` | Analysis scripts and their JSON results (see the table above). |
| `figures/` | Plotting scripts of Figures 3–5 (`plot_rq2.py`, `plot_java.py`, `plot_rq3.py`) and their shared style. |
| `experiments/knowledge_rag/` | Evidence-slot prompts (modes C, D, E): no retrieval, code pair, and knowledge; `next_stage/e_arm.py` (CWE/CVE entries), `next_stage/oracle_arm.py` (the pair's own fix and unrelated-pair evidence, the control experiment of RQ2), `next_stage/order_arm.py` (RQ3: reordered C, D, E and swapped demonstrations of A), `next_stage/shared_knowledge_arm.py`. |
| `experiments/knowledge_rag_deepseek/` | The same framework configured for DeepSeek-V4-Pro. |
| `experiments/vulrag_full/` | End-to-end Vul-RAG; `vendor/` holds the authors' code at a pinned commit. `detect_with_imported_kb.py` and `detect_swapped_knowledge.py` run detection with GLM-5.1 on GPT-5.5's knowledge base. |
| `experiments/java_rq1_server/`, `experiments/java_b_relabel/` | MegaVul retrieval for modes B and E (run on a GPU server; no model calls), the Java B prompts (`build_b_java_prompts.py`), the E inputs (`build_e_java_inputs.py`), NVD descriptions (`fetch_nvd_desc_java.py`), and the generated description queries (`java_vulrag_queries.py`). |
| `experiments/run_*.sh` | Model calls of the later experiments: one and four demonstrations (`run_a_k_sensitivity.sh`, `run_a_k1_gpt.sh`, `run_a_k4_gpt.sh`), the second GLM-5.1 run (`run_glm_repeat.sh`), the Java runs (`run_java_std.sh`, `run_java_be.sh`, `run_java_b_diff.sh`), and the RQ3 reorderings (`run_order_glm.sh`, `run_order_gpt.sh`). |
| `label/` | Two annotators' judgments of 51 sampled errors (earlier draft). |
| Top-level scripts | Chunk indexing and retrieval (`build_chunk_index.py`, `build_query_chunks_sim_rank.py`), patch contrast, prompt runners (`run_prompting_sliced_rag.py`, `icl/`, `Baseline/`), mode E corpora (`fetch_nvd_cwe.py`, `build_e_retrieval.py`), prompt replay for other models (`replay_saved_prompts.py`), context controls (`run_bias_controlled_prompt.py`), and output parsing (`evaluate_prompt_outputs.py`). |
| `tools/` | Dataset rebuilding (`build_labeled_test.py`, `build_megavul_java.py`), the export of `predictions/` from the full outputs (`export_compact_predictions.py`, `export_b_keys.py`), and its restoration (`expand_predictions.py`). |

## Inputs not included

- **Benchmark data.** PrimeVul: download the paired split from the PrimeVul authors;
  `tools/build_labeled_test.py` checks the test file against the SHA-256 of the version we used
  (`384758c2…`) and the labeled output against the file of the paper (`878134e6…`). MegaVul: download
  the Java part of release 2024-04 (`megavul.json`, linked from https://github.com/Icyrockton/MegaVul);
  `tools/build_megavul_java.py` checks it (`32bfb057…`) and the four files it builds.
- **Full run outputs**, which repeat the complete prompt, including the benchmark code, in every row.
  `predictions/` keeps everything the paper's analyses read; the run scripts regenerate the prompts.
- **API credentials and endpoints.** Model calls read the key and the OpenAI-compatible base URL from
  the environment: `SLICERAG_API_KEY` with `SLICERAG_BASE_URL` (evidence-slot runs) or
  `SLICERAG_OPENAI_BASE_URL` / `SLICERAG_ANTHROPIC_BASE_URL` (prompt runners). The run scripts take
  `SLICERAG_UNIAPI_BASE_URL` and `SLICERAG_UNIAPI_KEY` (the endpoint used for GLM-5.1, DeepSeek-V4-Pro,
  and the replayed prompts), `SLICERAG_NEWAPI_BASE_URL` and `SLICERAG_NEWAPI_KEY` (the endpoint used for
  the GPT-5.5 evidence-slot runs), and `SLICERAG_UNIAPI_HOST` (host name of the former, for routing).
  An experiment is always sent through the same endpoint as the prompt it is compared with.
  `SLICERAG` is the working name of the code base.

## Models and decoding

GPT-5.5 and GLM-5.1 run every usage mode; Claude Opus 4.7 and DeepSeek-V4-Pro
(`deepseek-v4-pro-0813`) run one configuration of each mode and the regrouped chunks and shortened CWE
entries of RQ3. GLM-5.1 and DeepSeek-V4-Pro requests use temperature 0 with thinking disabled;
GPT-5.5 (reasoning effort none) and Claude Opus 4.7 (no extended thinking) requests set no
temperature. The first unambiguous YES/NO in an answer is the prediction.

## Notes on the data

- Some GPT-5.5 answers first came back partly in Chinese, and we re-requested them with the same
  prompts and configuration. (a) Four of the 3,480 fixed-candidate detection answers began with a
  one-sentence plan in Chinese; `experiments/knowledge_rag/repeat_cjk_detection.py` re-requested
  exactly these four, and all four kept their YES/NO prediction
  (`outdir/knowledge_rag_gpt-5.5_cjk_repeat/comparison.json`); `predictions/`
  holds the merged answers. (b) Sixteen Vul-RAG knowledge-extraction calls (purpose, function,
  analysis, or knowledge step of 16 of the 3,721 training pairs) contained Chinese text; their cached
  calls were removed and re-requested (`reextraction_cjk.json`), retrieval was rerun, and detection
  was rerun for the 10 test functions whose retrieved knowledge changed (`redetect_rows.json`).
  The paper reports this run (`outdir/vulrag_gpt-5.5/`).
- The MegaVul Java line labels were first computed by matching the text of MegaVul's changed lines,
  which marks every line with the same text (for example each `}`). The paper uses labels computed as
  for PrimeVul (`relabel_java_diff.py`; applied to the PrimeVul test pairs, the procedure reproduces
  the stored labels of 860 of 870 functions), and mode B on the Java pairs was rerun with them
  (`outdir/java_std/*_B_chunks_diff`).
- Mode E code queries use CodeT5+ 110M embeddings, which need `transformers==4.46.3`; description
  queries use Sentence-BERT `all-mpnet-base-v2`.
