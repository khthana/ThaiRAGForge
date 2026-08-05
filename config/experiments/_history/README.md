# Historical one-off configs

These are `run --config` files used once to resume, patch, or rebuild a
specific slice of an index after a crash, a bug fix, or a corpus change.
Each was superseded by the time the next one appeared — none of them
should be re-run as-is today. They're kept (not deleted) because several
carry load-bearing narrative context in their header comments (e.g. the
ConGen/SCT `max_seq_length` truncation-bug fix) that `docs/*-log.md`
entries reference by filename.

For configs meant to be run today, see the parent directory
(`chunker_compare_full.yaml`, `entity_tags_full.yaml`, `rq3_*.yaml`,
`dev_smoke.yaml`, `chunker_compare_smoke.yaml`).
