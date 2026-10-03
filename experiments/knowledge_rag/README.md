# Fixed-candidate setting and CWE/CVE arms

`experiment.py` selects one historical vulnerable/fixed pair per test function by IDF-weighted
identifier overlap (from the training pairs that share no CVE, commit, or normalized function with
the test split), lets each model extract functional semantics, a root cause, and a fixing
condition from it, and runs four detection arms that share the target instruction and the
candidate: no retrieval, the complete code pair, all three knowledge fields, and knowledge
without the fixing condition. `concurrent_runner.py` sends the requests; `context_budget.py`
checks every request against the model's context limit before sending it, so no prompt is
truncated.

`next_stage/e_arm.py` fills the same evidence slot with the top three CWE entries (code query or
description query), CVE descriptions, or short CWE entries (identifier, name, first sentence);
the retrieval results come from `data/e_retrieval/`. `next_stage/run_e.sh MODEL ARM run 4` runs
one arm; `../run_e_until_complete.sh MODEL` repeats until every function is answered.

Configure the endpoint and key with `SLICERAG_BASE_URL` and `SLICERAG_API_KEY`. The `test_*.py`
files are offline unit tests (`python3 -m unittest discover -s experiments/knowledge_rag`).
