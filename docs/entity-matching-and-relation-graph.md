# Entity matching and the relation graph — full derivation

**This is the research record. The verdicts, bounds and operational rules live in
`CLAUDE.md`; this file is why they hold.** Folded out on 2026-08-23 from two
bullets totalling 22.5 KB, because `CLAUDE.md` is loaded into every session. The
text is **verbatim**, so every figure it carried stays under
`audit_doc_claims.py`'s D2/D5/D7 — this file is in `DOCS`.

**Why a new file rather than `docs/relation-graph.md`:** that one is *generated*
by `build_relation_graph.py` and would be overwritten on the next `--render`.
Same for `docs/program-matcher-absorption.md` and
`docs/program-tag-regeneration.md` — all three are reports, listed in
`audit_doc_claims.ARTIFACT_FILES` so their figures can be cited, and none of them
is a place to keep narrative.

---

- **Simple relation graph (`tools/corpus_prep/build_relation_graph.py` →
`data/graph/relations.json` + `docs/relation-graph.md`, 2026-08-10)** — the two
edges that need no new model, no GPU and no new dictionary: **A**
`program —belongs_to→ faculty` and **A′** `person —affiliated_with→ faculty`
(the inline `(สังกัดคณะX)` shape). Edges B (`person→responsible_for→program`)
and C (`person→replaces→person`) were out of scope pending a decisive measurement;
**that measurement is done and they are NOT being built (2026-08-10 — see the RQ4
entity-arms paragraph below for the numbers and the bound).** **No gold query
in either set is multi-hop, so this is a capability the current eval is
structurally unable to score — no retrieval gain may be claimed**, and every
denominator is an entity *dictionary*, itself a curated subset, so read the
coverage as "of what was found". Tags are recomputed from the tested matchers
rather than read from `academic_resolutions/entity_tags/*_by_file.json`, which
are dated 2026-07-17..25 and predate the person-loader fixes, the 07-28 OCR
remediation, the 08-08 title repair and the 08-09 re-OCR — building on them
would be this project's signature two-artifacts-from-different-days failure.
One corpus walk (~22 min, 2,854 files) caches its evidence to
`data/results/relation_graph_raw.json` so `--render` re-derives graph and report
free. **Edge A: 180 of 250 programs resolved, 56 ambiguous, 14 no_evidence**
— and the `ambiguous` bucket is **not one thing**: only **8** have two faculties
genuinely pointing at each other, the other **48** have a single faculty with
fewer witnesses than `min_votes=2`.
**The denominator is 250 and not 253 as of 2026-08-21, and the change is a
DEFINITION not a measurement — never read the two side by side as drift.**
`programs.json` holds 253 canonical names for 250 programmes: KOSEN renamed
three associate degrees in 2568 and both names must stay in the dictionary so
`match_programs` still matches documents written under either. Keyed on
*entries*, the graph counted those three twice **and split their evidence**,
which is not cosmetic — one half of the แมคคาทรอนิกส์ pair had a single witness
and sat in `ambiguous` while its twin resolved on four votes at the *same*
faculty. `build_graph` now iterates `programme_groups()`, pools each group's
votes onto its first entry in dictionary order and keeps the rest as `aka`.
The whole delta is those 6 entries (5 resolved + 1 ambiguous → 3 resolved), on
**byte-identical evidence** (the 08-11 cache), so this is not a fourth matcher
repair. **The check had to follow its subject**: `S2` gated
`len(records) == len(programs)` and would have FAILED on a correct graph, so it
now compares against the group count and prints both numbers
([[feedback_cleanup_can_break_an_audit]]); new **`S5`** gates that every one of
the 253 names is still reachable as a node or an `aka`, because grouping's real
risk is not a wrong merge (`programme_groups` is tested for that) but an entry
going **missing**, which would shrink every denominator here while every other
check still passed. This is also what made `programme_groups` a live function
rather than one only its own tests called. The §3 cross-checks are deliberately
**left per-name**: they ask whether two sources agree about one written name,
which is a different question from counting programmes.
**The pre-grouping figures are kept because the three walks below are stated
on them, and are superseded: 182 / 57 / 14 of 253 (split 8 / 49). They are
the SECOND 2026-08-11 re-walk's, and both re-walks were forced rather than
routine**: this script calls
`match_programs`, so each half of the matcher repair below moved the graph
without touching its own generator — the two-artifacts-from-different-days shape
again, now guarded by a `program_loader.py → relation-graph.md` edge in
`audit_doc_claims.py`'s `EVAL_INPUTS`. Both moves went in the predicted
direction: **170 / 60 / 23** (split 9 / 51) before either repair → **177 / 62 /
14** (12 / 50) after the degree half → **182 / 57 / 14** (8 / 49) after the
cross-subject half — all three on the 253-entry denominator, so compare them
with each other and not with the 250-programme figures above. A rescued tag returns evidence to the program that actually
owns it, which is why `no_evidence` fell by 9 on the first walk; a *dropped*
absorption stops lending a foreign programme's votes to its neighbour, which is
why `ambiguous` fell by 5 on the second. **The motivating case of the whole
audit visibly moved**: `หลักสูตรแพทยศาสตรบัณฑิต` — the row that held both
`ทันตแพทยศาสตรบัณฑิต` and `พยาบาลศาสตรบัณฑิต` — left the `ambiguous` table
altogether and is now `resolved → คณะแพทยศาสตร์` on 8 votes. The two
cross-checks moved too (see below) and all **five** self-checks still PASS. Two
distinct causes underneath, and the
first is the important one: **`program → faculty` is not a function** —
`วิศวกรรมเครื่องกล` really is offered by both `คณะวิศวกรรมศาสตร์` and
`วิทยาเขตชุมพรเขตรอุดมศักดิ์` (23 vs 18 votes), so any graph forcing one faculty
per program is wrong for that group by construction. The second is a matcher
finding worth its own ticket: **`match_programs` has no "matches nothing" exit
for a near-miss**, so a corpus name absent from the dictionary is absorbed by
its nearest neighbour — `หลักสูตรทันตแพทยศาสตรบัณฑิต` *and*
`หลักสูตรพยาบาลศาสตรบัณฑิต` both match `หลักสูตรแพทยศาสตรบัณฑิต`, which accounts
for that whole ambiguous row. Scope was **measured, not guessed**: 0 of 253
dictionary names collide with each other, so the collision is only with names
outside it. **Both halves are now REPAIRED (2026-08-11) — the degree half in
the paragraph after next, the same-degree cross-subject half (exactly the
dental/nursing row above) in the one after that.** `match_programs` is also read by
`build_gold_candidates.py` and `router`, so moving the threshold would move
published numbers. **That blast radius was MEASURED and it is zero on both
call sites (2026-08-10, `tools/eval/audit_program_matcher_absorption.py` →
`docs/program-matcher-absorption.md`, ~23 min walk cached so `--render` is
free).** Corpus-wide the defect is large — 9,141 accepted matches over 1,710
files, **23.1% (2,114) absorb a genuinely different name**, 210 of 249 matched
canonicals absorb at least one — and its dominant shape is the one worth
naming: **35.7% of absorptions swap the degree level** (บัณฑิต ↔ มหาบัณฑิต ↔
ดุษฎีบัณฑิต), one token apart so the ratio stays far above 0.82 while a
master's programme is tagged as the bachelor's of the same subject. **But it
reaches neither published path, and both were verified rather than assumed**:
`program_candidates()` never calls `match_programs` for membership — and it
never reads the *tags* either, which is stronger than this file used to say
(the old "seeds from tagged files" wording overstated the coupling, corrected
2026-08-12). It iterates `programs_by_file.json`'s **keys**, and
`tag_programs.py` writes a key for every live corpus file including the
zero-match ones, so the pool *is* the corpus and the only gate is `canonical
in resolution_id`, an exact substring of the manifest title. The matcher
therefore cannot move the program qrels **structurally**; only corpus
*membership* can. That is executed, not argued — `S2` in
`tools/corpus_prep/audit_program_tag_regeneration.py` blanks every value and
requires identical output — and the independent second route still holds:
**0 of 30** program queries' gold pairs
have a `resolution_id` failing to contain the program; and `classify_query`
asks only whether *any* program matched, never which one, so a name-for-name
swap **cannot** change a route (33/13/30/30 exact, 0 program queries routed
elsewhere).
**The degree-level swap was then REPAIRED (2026-08-11), and the useful part is
how the rule was chosen: three candidate rules were each walked over the whole
corpus, and both losers were rejected by their own numbers.** The naive
*reject* (drop the tag whenever the text's degree contradicts the winner's)
and *fallthrough* (take the next candidate) both assume the guard's job is to
decide keep-or-drop. It is not: of the **752** mentions where a winner's degree
is contradicted, **354 (47%) had a same-degree candidate whose subject the text
also supports already sitting in the dictionary** — the matcher had not run out
of options, it had **ranked** them wrong. So what ships is *select*: the degree
**filters the candidate set** and the best surviving candidate is re-selected,
which strictly dominates reject (**134 tags gained / 340 lost / 44 files
stripped bare**, against reject's 0 / 340 / 71). **The subject test inside the
rescue is load-bearing, not belt-and-braces**: 213 of the 752 have a
same-degree candidate that disagrees on subject, so filtering on degree alone
would hand `...ดุษฎีบัณฑิต ...ไฟฟ้า` to `...โยธา` — trading a wrong degree for a
wrong subject. The remaining 129 are **undecidable** (one side has no
`สาขาวิชา`, so no evidence means no rescue, [[feedback_undefined_is_not_zero]])
and 56 have nothing at the text's own degree, where *matches nothing* is now
finally an available answer. **Blast radius re-measured after the repair, not
inherited**: 0 of 247 published program gold pairs move and the router stays
33/13/30/30 — which was a counterfactual when it was written, because
`build_gold_candidates.py` reads the **cached**
`academic_resolutions/entity_tags/programs_by_file.json` (2026-07-25), so
nothing published moved until that artifact was regenerated. **It was
regenerated 2026-08-12 and the counterfactual is now closed as a real
measurement** — see the regeneration bullet below; the 0 and the 33/13/30/30
both reproduce against the new artifact.
**The same-degree cross-subject half — the dental/nursing shape the degree
filter is structurally unable to see — was then CLOSED the same day, and the
mechanism is the reusable part: dilution by concatenation
([[feedback_a_similarity_over_a_concatenation_dilutes]]).** The ratio runs
over head noun + subject *joined*, so a disagreement confined to one half is
averaged away by the other half agreeing — `ทันตแพทยศาสตรบัณฑิต` against
`แพทยศาสตรบัณฑิต` scores **0.88**, comfortably over the 0.82 threshold, purely
because the shared `แพทยศาสตรบัณฑิต` outweighs the `ทันต` prefix. Testing the
two halves separately (`_head_contradicted` / `_subject_contradicted`) removes
the dilution. **The two halves need different tests, and the asymmetry is
structural, not a fudge**: `_bounded_span_for` sizes the window from the match
position, so the head noun is always covered in full while the subject sits at
the tail and is routinely truncated — truncation is therefore *forgiven* in
the subject (longest-common-substring coverage, so a cut name still scores
1.00) and **not** forgiven in the head (extra leading material is a different
formal degree name). Result: the guard fires on **606** of 9,141 accepted
matches and is mostly **drops** (12 rescued, 594 dropped: 159 head, 435
subject) — which is itself the finding, because where a degree swap usually
has the right answer one level away in the dictionary, a cross-subject
absorption usually does not; the absorbed programme is simply **not in
`programs.json`**, and *matches nothing* is the right answer. Per file, both
guards together: **140 gained / 594 lost / 115 stripped bare / 446 changed**.
Blast radius unchanged and re-verified rather than inherited (§4 **0 of 30**
program gold queries, §5 router **33/13/30/30**), and the degree half's own
figures reproduce **exactly** (752 → 354/213/129/56), which is why the two are
reported as separate rows and never merged. **Two rules were added after the
first cross-subject walk, both because a self-check went red — the corpus was
fine and the guard was wrong, twice.** (1) **A drop needs *both* subject tests
against it.** Coverage is blind to a difference spread through the string
rather than sitting at one end, so a one-character OCR/dictionary variant
(`วิศวกรรมเล็กทรอนิกส์` vs `วิศวกรรมอิเล็กทรอนิกส์`) covers 0.55 while
`_field_agrees` — this project's own settled test for the same relation — puts
it at 0.95. **One pair must not be agreeing and contradicting at once**, so a
drop now needs coverage **and** ratio against it; **133 of 569** subject drops
sat in exactly that band ([[feedback_agreement_and_contradiction_are_one_relation]]). (2) **A cross-subject rescue may not cross the
degree**: that branch is reached precisely when the *winner's* degree is
uncontradicted, which says nothing about the *runner-up's*, and 6 rescues in
the first walk moved a mention between บัณฑิต and มหาบัณฑิต — a cross-subject
rescue silently undoing the settled degree rule (now `S9`). **The unit test for
(2) had to be taken from the corpus, not invented**: two brute-force searches
over degree/subject/tail grids returned **0** synthetic fixtures, because a
candidate that agrees on the subject is normally textually closer than the
contradicted winner and simply wins outright — the real mention
(`2564/ครั้งที่ 11`, where `<br/>` markup inflates each candidate's window
differently) is what reaches the branch. **What is still NOT fixed**: nothing
in either guard gives `match_programs` a *lexical* notion of subject identity,
so a subject the window truncates past recognition is still undecidable rather
than wrong. **The measurement that decided all this was
itself wrong once and the correction is the reusable part**
([[feedback_a_guards_precondition_biases_its_own_test]]): the first A/B asked
"does the text support the **subject** of the tag the guard removed?" and
answered 573 loss / 55 fix — an instrument that cannot return the other answer,
because every firing has a contradicted *degree* by construction, so the
subject is always the half that agreed. **Its first
run was wrong in a way worth remembering**: it reported "99.3% absorb a
foreign name" with the inserted-character distribution's mode at exactly
**4** — which is `_WINDOW_SLACK`, i.e. it was counting the matcher's own
read-ahead window as absorbed text
([[feedback_a_mode_on_a_constant_is_your_instrument]]). `S5` now pins that a
pure window tail scores 0 (6,333 such spans) while a longer tail still
registers its excess. **And the same trap caught the repair's own check**:
`S7` (every rescue agrees with the span's subject) FAILED on **4 of 354**
because it judged the rescue against the span sized to the **winner's** length,
while `_bounded_span_for` sizes the window to *each candidate*, so a rescue
onto a longer sibling name (`...การจัดการโลจิสติกส์` →
`...การจัดการโลจิสติกส์และซัพพลายเชน`) read further into the text than the
check let it. The record now carries `selected_span`, and the 4 are reported
as a fact about the dictionary (`S8`) instead of a violation that never
happened — **diagnose a red self-check from the cache before editing either
the rule or the check**; both times here the instrument was wrong and the
corpus was fine. **`entity_tags_full` was deliberately NOT rebuilt at first,
and was then rebuilt 2026-08-12 together with the cached artifact — never
alone, and that coupling is the rule to keep.** `entity_loader.py` is a third
call site, so leaving the index at its 2026-08-05 build held pre-repair program
tags in front of `entity_lookup`/`entity_boost` and the published RQ4 entity
arms, while rebuilding it *alone* would have decoupled its tags from the qrels'
own 2026-07-25 cached tags in an unmeasured way. So both moved together:
`programs_by_file.json` regenerated (see the bullet below) and the index
rebuilt from it (71,073 chunks, `docset_hash 7a274096d8609f61`), then both
entity result sets re-scored. **Only the program-bearing rows moved, which is
the built-in control** — the person/course/faculty loaders were untouched, and
under `entity_lookup` (pure set membership) their scores are identical to 4
decimals while `program` goes 0.8918 → **0.9013** and overall 0.9422 →
**0.9449**; `entity_boost` `program` recall@10 0.5765 → **0.5834**, with
person/course moving ±0.007 in *both* directions because a tag line is part of
chunk *text*, so changed program tags perturb the embeddings and BM25 of the
same documents even for a person query. **This refuted a pre-registered
prediction**: recall was predicted to fall, since the cross-subject guard cuts
far more tags than it adds (594 vs 140) — but the degree guard *re-selects* a
same-degree candidate instead of merely dropping, so a rescued tag lands on the
programme that actually owns it. The gating verdict in
[[project_rq4_entity_arms_gating]] was **re-measured, not inherited** (RQ4
entity arms re-run 2026-08-12 — only the cells whose context actually changed
were regenerated, the rest frozen byte-for-byte, per
`docs/rq4-design.md`): it is
unchanged, resting on **−0.2523** at Holm 0.0000, a *ranking* failure over
already gold-dense contexts (**0.6501**), which better program tags were not
expected to close — and did not. Rebuild it only together with a
regeneration of `programs_by_file.json`, and re-run the RQ4 entity arms if you
do. **Edge A′ is ~7x smaller than the scan note claimed and the
note is corrected**: `สังกัดคณะ` was recorded as appearing in "1,465 files
(51%)"; direct counting over the 2,854 live files gives `สังกัด` in **209**
files and `สังกัดคณะ` in **73**, and no alternative anchor supplies a larger
person→faculty source (`จากคณะ` 1,083 is `คณะกรรมการ` boilerplate). Anchored
extraction accepts 206 of 435 marker occurrences, attaches a person to 71, and
yields **66 people with an edge across 86 files (3.0%)**. **Read its
`resolved` 100% as "nothing contradicts it", not as quality** — 62 of 66 rest
on a single witness. **The finding that matters more than the count is what the
marker *means*, and it was measured rather than assumed**: the parenthetical is
written **100% of the time (64 of 64 determinable) for a person from a
*different* faculty than the document's own**, i.e. it is a cross-appointment
disambiguator. So A′ is a **biased** sample of people, and deriving
person→faculty from plain document co-occurrence the way edge A does would be
wrong *with a direction*, not merely noisy. Two independent cross-checks on A,
reported and never gated: manifest title vs OCR'd body agree on **170 of 181**
programs both can name (two independent text sources — typed vs scanned), and a
split-half over disjoint document sets agrees on **105 of 115**. Read them
across all three walks rather than as a level, because the two repairs pull the
denominator in opposite directions and neither is a quality signal on its own:
manifest-vs-body 158/169 → 165/180 → 170/181 (93.5% → 91.7% → **93.9%**),
split-half 103/112 → 110/118 → 105/115 (92.0% → 93.2% → **91.3%**). The degree
half *rescued* tags so both denominators grew; the cross-subject half is mostly
*drops*, so the split-half denominator fell 118 → 115 — a programme no longer
named in both halves is one the guard stopped inventing, which is the intended
effect and not a loss of coverage. Agreement stayed inside ~2 points throughout.
Four self-checks
(S1 every faculty node is canonical, S2 `no_evidence` stays *undefined* rather
than a low score, S3 a window-extracted faculty must also appear in the
document's own tags, S4 every A′ name must be one `find_people` found in that
same file) all pass; **S1 first reported a false FAIL from operator precedence**
(`{a} | {b} - dict` parses as `{a} | ({b} - dict)`), and **S2's first version was
vacuous** because the graph is built by iterating the dictionary, so "the buckets
add up" was true by construction — it now gates on the buckets staying
*distinguishable*. `docs/relation-graph.md` is in `audit_doc_claims.py`'s
`ARTIFACT_FILES` so its figures can be cited in prose.
- **`programs_by_file.json` regeneration, and the counterfactual it closed
(2026-08-12, `tools/corpus_prep/audit_program_tag_regeneration.py` →
`docs/program-tag-regeneration.md`).** The cached tags dated **2026-07-25**;
`match_programs` was repaired 08-11 and the corpus moved three times in
between, so the bullet above could only ever say "nothing published moves
*until that artifact is regenerated*". It is regenerated, and the useful part
is that **drift and repair were separated instead of reported as one delta**.
Overall 07-25 → 08-12 is 500 files changed, **+212 / −617** tags, 4,743 →
4,338; but that splits into **(B) corpus drift** (old matcher, new corpus:
109 files, +95/−46) and **(A) the matcher repair** (new corpus, old → new
matcher: 446 files, +140/−594), and the (A) row **reproduces
`docs/program-matcher-absorption.md`'s own per-file figures exactly** (S3) —
which is what licenses reading (B) as drift rather than as a residue of a
mis-modelled repair. The 08-08 title repair cannot appear in either row and
that is a fact about the pipeline, not an omission: `tag_programs.py` keys on
the **file path** and matches over **body text**, so a title change touches
neither. **The blast-radius claim got stronger, not merely confirmed**:
`program_candidates()` iterates the mapping's **keys** and never reads a tag
*value*, so the matcher cannot move the program qrels **structurally** — and
that is *executed*, not argued (S2 blanks every value and requires identical
output: 147 candidates either way). Measured anyway, both arms agree
147/662 with 0 moved; **0 of 247** scored program gold pairs lost; router
**33/13/30/30**. Two triage rules worth keeping. (1) **Diff against the
artifact that is actually published**: 1 candidate differs from
`gold_candidates.json`, which is a **superseded intermediate** (07-25,
regenerated 07-30 for the `resolution_id` fix) — the truncated
`2567/1/…(หลักสูตรนานาชาติ)` id became the two full `…๒๕๖๓`/`๒๕๖๔` ids and
`gold_query_set_73det.yaml` already holds both, so S6 gates on *scored pairs*
rather than on that file. (2) **Corpus membership is the only channel that
can move a candidate**, so S4 watches it: the corpus gained exactly one file
(`2568/ครั้งที่ 7/เรื่อง รับรองรายงานการประชุม.md`, the 08-09 CHECO
restoration), it matches zero programs and its `resolution_id` holds no
program canonical, which is why membership moved and the qrels did not.
`docs/program-tag-regeneration.md` is in `audit_doc_claims.py`'s
`ARTIFACT_FILES`, and `program_loader.py` now names it as a third `EVAL_INPUTS`
consumer — the whole report is a function of the matcher, so a future repair
turns it into a record of a matcher that no longer exists with nothing else on
disk saying so.
