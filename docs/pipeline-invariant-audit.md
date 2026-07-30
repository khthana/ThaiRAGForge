# Pipeline invariant audit

Run 2026-07-30 03:47 UTC. 19 pass / 5 warn / 1 fail.

| check | status | detail |
|---|---|---|
| C1 resolution_id unique | PASS | 2853 files -> 2853 ids |
| C2 no empty document | PASS | 0 empty of 2853 |
| C3a manifest entries point at real files | PASS | 0 dead |
| C3b no duplicate file keys | PASS | 0 duplicated |
| C3c every corpus file listed in its manifest | PASS | 0 unlisted (title falls back to filename stem) |
| C3d one source URL, one base title | PASS | 0 URLs claimed by differently-titled documents |
| C4 no orphaned .md.dup archive | WARN | 24 of 240 archives have neither a live file nor split pieces |
| C5 master_list.csv row count | PASS | 2853 rows vs 2853 corpus files (rows are meeting-level, not file-level -- informational) |
| I1 row alignment (chunks/vectors/lexical) | PASS | 0 misaligned of 55 |
| I2 chunk_id unique within index | PASS | 0 indexes with duplicates |
| I3a no chunks from outside the corpus | PASS | 0 indexes contaminated |
| I3b corpus coverage | PASS | lowest: chunker_compare_full 2853/2853, chunker_compare_full 2853/2853, chunker_compare_full 2853/2853, chunker_compare_full 2853/2853, chunker_compare_full 2853/2853 |
| I4 embeddings finite and non-zero (sampled) | PASS | 0 indexes with bad vectors |
| I5 manifest n_resolutions matches corpus | PASS | 0 drifted |
| I6 index newer than corpus | PASS | 0 indexes built before the corpus's last edit |
| E1 gold ids resolve (gold_query_set_73det.yaml) | PASS | 1046 refs, 0 dangling |
| E2 no duplicate query (gold_query_set_73det.yaml) | PASS | 106 queries, 0 duplicated |
| E1 gold ids resolve (gold_query_set.yaml) | PASS | 1219 refs, 0 dangling |
| E2 no duplicate query (gold_query_set.yaml) | WARN | 252 queries, 5 duplicated |
| E0 combo id identifies its index unambiguously | FAIL | 12 combo ids exist under >1 index root (BuildCombo.id omits the corpus, so results cannot be attributed to one index) |
| E3a results reference ids their index holds | PASS | 0 result files with unknown ids |
| E3d retired result sets name ids no index holds | WARN | 346 result files in ['congen_sct_truncation_fix', 'gold_chunker_compare', 'gold_chunker_compare_73det', 'gold_embedder_compare', 'gold_full_embedder_matrix', 'mode_b_routed', 'silver_chunker_compare'] carry titles from an earlier corpus state -- retired, not read by any current script |
| E3c retired result sets cite pre-fix contamination ids | WARN | 2762 result files cite an id from the corpus-discovery contamination bug -- expected for result sets computed before its fix; do not reuse them |
| E3b results answer a known gold query | WARN | 1022 unrecognized queries |
| E4 results newer than their index | PASS | 0 result sets computed before their index was rebuilt |
