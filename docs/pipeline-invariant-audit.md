# Pipeline invariant audit

Run 2026-08-08 14:50 UTC. 24 pass / 0 warn / 1 fail.

| check | status | detail |
|---|---|---|
| C1 resolution_id unique | PASS | 2854 files -> 2854 ids |
| C2 no empty document | PASS | 0 empty of 2854 |
| C3a manifest entries point at real files | PASS | 0 dead |
| C3b no duplicate file keys | PASS | 0 duplicated |
| C3c every corpus file listed in its manifest | PASS | 0 unlisted (title falls back to filename stem) |
| C3d one source URL, one base title | PASS | 0 URLs claimed by differently-titled documents |
| C4 no orphaned .md.dup archive | PASS | 0 of 239 archives are unaccounted for (21 are tail fragments of a live file's title, 1 are live under a repaired name, 1 under a truncated title, 0 under a title naming a different agenda item) |
| C5 master_list.csv row count | PASS | 2854 rows vs 2854 corpus files (rows are meeting-level, not file-level -- informational) |
| I1 row alignment (chunks/vectors/lexical) | PASS | 0 misaligned of 55 |
| I2 chunk_id unique within index | PASS | 0 indexes with duplicates |
| I3a no chunks from outside the corpus | PASS | 0 indexes contaminated |
| I3b corpus coverage | PASS | lowest: chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854 |
| I4 embeddings finite and non-zero (sampled) | PASS | 0 indexes with bad vectors |
| I5 manifest n_resolutions matches corpus | PASS | 0 drifted |
| I6 index newer than corpus | PASS | 0 indexes built before the corpus's last edit |
| E1 gold ids resolve (gold_query_set_73det.yaml) | PASS | 1046 refs, 0 dangling |
| E2 no duplicate query (gold_query_set_73det.yaml) | PASS | 106 queries, 0 duplicated |
| E1 gold ids resolve (gold_query_set.yaml) | PASS | 1219 refs, 0 dangling |
| E2 no duplicate query (gold_query_set.yaml) | PASS | 252 queries, 0 duplicated |
| E0 combo id identifies its index unambiguously | FAIL | 12 combo ids exist under >1 index root (BuildCombo.id omits the corpus, so results cannot be attributed to one index) |
| E3a results reference ids their index holds | PASS | 0 of 23156 live result files reference an id their index does not hold |
| E3d retired result sets name ids no index holds | PASS | 0 of 0 retired result files carry titles from an earlier corpus state (retired sets, read by no current script; 0 of 0 means they have been archived off-repo, not that they were checked) |
| E3c retired result sets cite pre-fix contamination ids | PASS | 0 of 23156 result files cite an id from the corpus-discovery contamination bug -- expected for sets computed before its fix; do not reuse them |
| E3b results answer a known gold query | PASS | 0 unrecognized queries across 23156 result files (['mode_b', 'mode_b_routed'] excluded: interactive UI queries are not gold by design) |
| E4 results newer than their index | PASS | 0 result sets computed before their index was rebuilt |
