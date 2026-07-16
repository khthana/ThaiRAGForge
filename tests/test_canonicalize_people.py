"""Pure-logic tests for canonicalize_people.py: name aggregation, the two
merge rules, and clustering. No corpus I/O -- collect_raw_counts's
file-walking loop is exercised manually against the real corpus, same
convention as the rest of tools/corpus_prep/ (see test_tag_people.py).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import canonicalize_people as canon  # noqa: E402


class TestAggregateByName:
    def test_collapses_title_out_of_the_identity_key(self):
        # the same person promoted from ผศ. to รศ. between meetings must not
        # split into two clusters -- title is dropped from the key entirely
        raw = Counter(
            {
                ("ผศ.ดร.", "สมชาย", "ใจดี"): 5,
                ("รศ.ดร.", "สมชาย", "ใจดี"): 3,
            }
        )
        name_counts, title_counts = canon.aggregate_by_name(raw)
        assert name_counts == {("สมชาย", "ใจดี"): 8}
        assert title_counts[("สมชาย", "ใจดี")] == Counter({"ผศ.ดร.": 5, "รศ.ดร.": 3})


class TestPrefixRelated:
    def test_dropped_trailing_syllable_is_prefix_related(self):
        assert canon._prefix_related("ศิริพันธ์โนน", "ศิริพันธ์") is True

    def test_unrelated_short_strings_are_not_merged_via_prefix(self):
        assert canon._prefix_related("กก", "กกกกกกกก") is False  # too short

    def test_a_common_short_surname_prefix_does_not_bridge_unrelated_names(self):
        # "ศิริ" is a common Thai surname-starting syllable shared by many
        # genuinely different surnames -- a 4-char stub of a 12-char surname
        # retains only 33%, nowhere near a plausible single dropped
        # syllable, so it must not be treated as a truncation
        assert canon._prefix_related("ศิริพันธ์โนน", "ศิริ") is False

    def test_low_retention_ratio_is_not_merged_even_within_a_short_gap(self):
        assert canon._prefix_related("ก" * 4, "ก" * 10) is False  # 40% retained


class TestClusterPeople:
    def test_merges_a_truncated_surname_variant(self):
        name_counts = {
            ("ปุณณมา", "ศิริพันธ์โนน"): 611,
            ("ปุณณมา", "ศิริพันธ์"): 32,
        }
        title_counts = {
            ("ปุณณมา", "ศิริพันธ์โนน"): Counter({"รศ.ดร.": 611}),
            ("ปุณณมา", "ศิริพันธ์"): Counter({"รศ.ดร.": 32}),
        }
        entries = canon.cluster_people(name_counts, title_counts)
        assert len(entries) == 1
        assert entries[0]["canonical_surname"] == "ศิริพันธ์โนน"
        assert entries[0]["count"] == 643
        assert entries[0]["aliases"] == [
            {"given": "ปุณณมา", "surname": "ศิริพันธ์", "count": 32}
        ]

    def test_merges_a_misread_leading_character_in_given_name(self):
        name_counts = {
            ("ถิรายุ", "ชุมสาย"): 151,
            ("ภิรายุ", "ชุมสาย"): 85,
        }
        title_counts = {
            ("ถิรายุ", "ชุมสาย"): Counter({"ผศ.ดร.": 151}),
            ("ภิรายุ", "ชุมสาย"): Counter({"ผศ.ดร.": 85}),
        }
        entries = canon.cluster_people(name_counts, title_counts)
        assert len(entries) == 1
        assert entries[0]["canonical_given"] == "ถิรายุ"
        assert entries[0]["count"] == 236

    def test_does_not_merge_two_distinct_people(self):
        name_counts = {
            ("สมชาย", "ใจดี"): 10,
            ("วิชัย", "รักเรียน"): 5,
        }
        title_counts = {
            ("สมชาย", "ใจดี"): Counter({"ดร.": 10}),
            ("วิชัย", "รักเรียน"): Counter({"ดร.": 5}),
        }
        entries = canon.cluster_people(name_counts, title_counts)
        assert len(entries) == 2
        assert all(e["aliases"] == [] for e in entries)

    def test_transitively_merges_through_a_shared_variant(self):
        # A and C aren't directly similar enough to merge on their own, but
        # both merge with B -- union-find should still put all three in one
        # cluster (A~B via prefix, B~C via given-name fuzz)
        name_counts = {
            ("ปุณณมา", "ศิริพันธ์โนน"): 611,  # A
            ("ปุณณมา", "ศิริพันธ์"): 32,  # B (surname prefix of A)
            ("ปุณณมข", "ศิริพันธ์"): 1,  # C (given name fuzzy-similar to B)
        }
        title_counts = {k: Counter({"รศ.ดร.": v}) for k, v in name_counts.items()}
        entries = canon.cluster_people(name_counts, title_counts)
        assert len(entries) == 1
        assert entries[0]["count"] == 644

    def test_canonical_spelling_is_the_most_frequent_variant(self):
        name_counts = {("อรัญญา", "วลัยรัชต์"): 450, ("อรัญญา", "วลัยรัช"): 28}
        title_counts = {k: Counter({"ผศ.ดร.": v}) for k, v in name_counts.items()}
        entries = canon.cluster_people(name_counts, title_counts)
        assert entries[0]["canonical_surname"] == "วลัยรัชต์"

    def test_a_common_short_prefix_stub_does_not_bridge_two_real_people(self):
        # regression: "ศิริ" alone used to bridge "ศิริพันธ์โนน" and
        # "ศิริพงศ์" (two different, unrelated surnames) into one cluster
        # via union-find transitivity, since it was within the old flat
        # gap tolerance of both
        name_counts = {
            ("ปุณณมา", "ศิริพันธ์โนน"): 611,
            ("ปุณณมา", "ศิริ"): 2,
            ("ปุณณมา", "ศิริพงศ์"): 1,
        }
        title_counts = {k: Counter({"รศ.ดร.": v}) for k, v in name_counts.items()}
        entries = canon.cluster_people(name_counts, title_counts)
        surnames_by_cluster = [
            {e["canonical_surname"], *(a["surname"] for a in e["aliases"])}
            for e in entries
        ]
        assert not any(
            "ศิริพันธ์โนน" in group and "ศิริพงศ์" in group
            for group in surnames_by_cluster
        )

    def test_canonical_title_is_the_most_frequent_across_all_variants(self):
        name_counts = {("สมชาย", "ใจดี"): 8}
        title_counts = {
            ("สมชาย", "ใจดี"): Counter({"ผศ.ดร.": 3, "รศ.ดร.": 5})
        }
        entries = canon.cluster_people(name_counts, title_counts)
        assert entries[0]["canonical_title"] == "รศ.ดร."
