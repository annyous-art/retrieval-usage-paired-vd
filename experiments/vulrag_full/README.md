# Vul-RAG, end to end

Runs Vul-RAG with the authors' code (`vendor/`, pinned commit and file hashes in
`vendor/provenance.json`, license in `vendor/LICENSE`) on the PrimeVul paired split with GPT-5.5,
using the repository's documented settings: the three query elements (abstract purpose, detailed
behavior, code) retrieve 20 candidates each, fused by rank sum; detection checks at most three
knowledge items one at a time and stops at the first decisive one, and a function without a
decisive item is predicted NO, as in the official evaluator. The knowledge base is extracted from
the 3,721 training pairs that share no CVE, project and commit, or normalized function with the
test split.

`run.py` prepares, extracts, retrieves, and detects; `data_utils.py`, `prompts.py`, and
`retrieval.py` adapt the vendor code to the paired split; `test_offline.py` runs without API
calls. The outputs used in the paper are in `outdir/vulrag_gpt-5.5/`.
`detect_with_imported_kb.py` reuses that run's knowledge base and query descriptions and runs only
retrieval and detection with GLM-5.1 (`outdir/vulrag_glm-5.1_gpt55_knowledge/`);
`detect_swapped_knowledge.py` then gives each target the knowledge retrieved for another test pair
(fixed derangement, seed 20260915; `outdir/vulrag_glm-5.1_other_pair_knowledge/`). Set the
endpoint with `SLICERAG_NEWAPI_BASE_URL` (or `SLICERAG_UNIAPI_BASE_URL` with `--provider uniapi`)
and the key with `SLICERAG_API_KEY`.
