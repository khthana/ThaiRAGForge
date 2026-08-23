# Pipeline invariant audit

Run 2026-08-23 14:29 UTC. 31 pass / 0 warn / 1 fail.

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
| T1 cached entity tags reproduce from the current matchers | FAIL | files differing of files compared, over a 60-file sample: people 17/60, programs 0/60, courses 2/60, faculties 0/60; build_gold_candidates.py (the qrels generator) reads people/courses/faculties VALUES and now refuses on this, so nothing can silently derive qrels from them. RE-RUNNING tag_*.py IS NOT AN OBVIOUS FIX and is a decision, not a chore: it would make these copies current with today's corpus while data/index/entity_tags_full still holds tags from its own build date, i.e. it moves the mismatch rather than removing it, and CLAUDE.md coupled that index to a tag regeneration for exactly this reason. Decide the pair together |
| T1b a programs drift cannot move the qrels | PASS | program_candidates() iterates the tag mapping's KEYS and never reads a value -- audit_program_tag_regeneration.py S2 blanks every value and requires identical output -- so a programs difference is a reporting concern, not a qrels one |
| I1 row alignment (chunks/vectors/lexical) | PASS | 0 misaligned of 55 |
| I2 chunk_id unique within index | PASS | 0 indexes with duplicates |
| I3a no chunks from outside the corpus | PASS | 0 indexes contaminated |
| I3b corpus coverage | PASS | lowest: chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854, chunker_compare_full 2854/2854 |
| I4 embeddings finite and non-zero (sampled) | PASS | 0 indexes with bad vectors |
| I5 manifest n_resolutions matches corpus | PASS | 0 drifted |
| I6 index newer than corpus | PASS | 0 indexes built before the corpus's last edit |
| I7 index matches the build its writer sealed | PASS | 0 unsealed, 0 mismatching of 55 (fix: python tools/seal_index_dirs.py --apply) |
| I8 one corpus state per (loader, n_resolutions) | PASS | 5 (loader, n_resolutions) groups over 55 indices, 0 holding more than one docset_hash; largest 39; 2 group(s) of one are UNWATCHED here (entity_tags/2854, normalized/2854) -- I6/I7 cover those |
| E1 gold ids resolve (gold_query_set_73det.yaml) | PASS | 1046 refs, 0 dangling |
| E2 no duplicate query (gold_query_set_73det.yaml) | PASS | 106 queries, 0 duplicated |
| E1 gold ids resolve (gold_query_set.yaml) | PASS | 1219 refs, 0 dangling |
| E2 no duplicate query (gold_query_set.yaml) | PASS | 252 queries, 0 duplicated |
| E0 every result attributes to exactly one index | PASS | 0 of 23165 result files cannot be attributed to one built index (12 of 43 combo ids exist under >1 index root, since BuildCombo.id omits the corpus; attributed 4 by elimination, 850 by no built index, 22305 by recorded, 6 by unique name) |
| E3a results reference ids their index holds | PASS | 0 of 23163 live result files reference an id their index does not hold |
| E3d retired result sets name ids no index holds | PASS | 0 of 2 retired result files carry titles from an earlier corpus state (retired sets, read by no current script; 0 of 0 means they have been archived off-repo, not that they were checked) |
| E3c retired result sets cite pre-fix contamination ids | PASS | 0 of 23165 result files cite an id from the corpus-discovery contamination bug -- expected for sets computed before its fix; do not reuse them |
| E3b results answer a known gold query | PASS | 0 unrecognized queries across 23165 result files (['mode_b', 'mode_b_routed'] excluded: interactive UI queries are not gold by design) |
| E4 results newer than their index | PASS | 0 result sets computed before their index was rebuilt |
| G1a no RQ4 answer generated from a truncated prompt | PASS | 0 truncated of 2010 answers carrying num_ctx |
| G1b no pre-fix RQ4 answer is provably truncated | PASS | 0 truncated of 852 pre-fix answers carrying provable evidence either way about their prompt, none of it needing them regenerated (408 by the UTF-8-byte upper bound, 444 by a cached probe at num_ctx=8,192) |
| G1c every RQ4 answer's prompt fit is established | PASS | 0 of 2862 answers are unmeasured -- every published RQ4 answer's prompt is now established by a recorded field, the UTF-8-byte bound, or a probe at num_ctx=8,192 (data/results/rq4_prompt_fit_probes.md), never by the 0.95 screen |
