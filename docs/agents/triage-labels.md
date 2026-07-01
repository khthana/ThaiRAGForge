# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to
the actual label strings used in this repo's issue tracker. This repo uses the default
names (identity mapping).

| Canonical role     | Label in our tracker | Meaning                                  |
| ------------------ | -------------------- | ---------------------------------------- |
| `needs-triage`     | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`       | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`  | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`  | `ready-for-human`    | Requires human implementation            |
| `wontfix`          | `wontfix`            | Will not be actioned                     |

Category roles use GitHub's default labels: `bug` and `enhancement`.

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the
corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use. State labels
other than `ready-for-agent` do not yet exist in the repo; the `triage` skill will
create them on first use (or create them ahead of time with `gh label create`).
