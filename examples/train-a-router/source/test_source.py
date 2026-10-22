"""Small prospective source-family and allocation tests; no model outcomes."""

from collections import Counter
import unittest

from grouping import families
from prepare import ROLES, allocate, duplicate_key, partition


def row(identifier, task, request, target="unused"):
    return {
        "id": identifier,
        "task": task,
        "index": 0,
        "source_id": identifier,
        "source_path": "synthetic/task.json",
        "source_metadata": {},
        "request": request,
        "target": target,
    }


class SourceTests(unittest.TestCase):
    def test_linguini_shared_problem_but_not_common_format(self):
        rows = [
            row(
                "a",
                "linguini",
                "These are translations in the Kewa language of an original problem.\n\nTranslate apple.",
            ),
            row(
                "b",
                "linguini",
                "These are translations in the Kewa language of an original problem.\n\nTranslate pear.",
            ),
            row("c", "linguini", "These are translations in the Tonga language of another problem.\n\nTranslate pear."),
        ]
        identity, _, _ = families(rows)
        self.assertEqual(identity["a"], identity["b"])
        self.assertNotEqual(identity["a"], identity["c"])

    def test_causal_specific_story_variants_link(self):
        rows = [
            row(
                "a",
                "causal_understanding",
                "Question: Janet works in the factory that assembles complex electronic circuit boards for large cars. She presses a switch. Did it fail? Reply Yes or No.",
            ),
            row(
                "b",
                "causal_understanding",
                "Question: Janet works in a factory that assembles complex electronic circuit boards for large cars. She does not press a switch. Did it fail? Reply Yes or No.",
            ),
        ]
        identity, _, _ = families(rows)
        self.assertEqual(identity["a"], identity["b"])

    def test_reused_time_primitive_does_not_link_compositions(self):
        common = "Question1: What date follows 2024-01-01?\n\n"
        rows = [
            row("a", "time_arithmetic", common + "Let X = X + 2.\n\nQuestion2: What is X o'clock in Paris?"),
            row("b", "time_arithmetic", common + "Let X = X + 3.\n\nQuestion2: What is X o'clock in Paris?"),
        ]
        identity, _, coverage = families(rows)
        self.assertNotEqual(identity["a"], identity["b"])
        self.assertTrue(all(coverage.values()))

    def test_same_sarc_post_with_different_replies_is_not_a_family(self):
        def request(suffix):
            return "\n".join(
                f"POST {i}: This is a lengthy common post number {i}. REPLY {i}: This is different response {suffix} {i}."
                for i in range(1, 4)
            )

        identity, _, _ = families(
            [row("a", "sarc_triples", request("first")), row("b", "sarc_triples", request("second"))]
        )
        self.assertNotEqual(identity["a"], identity["b"])

    def test_family_identity_does_not_depend_on_reference(self):
        rows = [
            row("a", "multistep_arithmetic", "Consider this complete composed arithmetic problem with two operators."),
            row(
                "b", "multistep_arithmetic", "Consider this different composed arithmetic problem with three operators."
            ),
        ]
        before = families(rows)[0]
        for item in rows:
            item["target"] = "changed reference that grouping must not read"
        self.assertEqual(before, families(rows)[0])

    def test_allocation_is_stable_whole_family_and_proportional(self):
        groups = [f"family-{index}" for index in range(180)]
        assigned = allocate("task", groups)
        self.assertEqual(assigned, allocate("task", list(reversed(groups)) + groups[:4]))
        self.assertEqual(Counter(assigned.values()), {"train": 90, "development": 27, "calibration": 18, "final": 45})

    def test_mini_reserves_context_family_and_conflict_excludes_whole_family(self):
        rows = [
            row(
                str(i),
                "linguini",
                f"Original independent language problem introduction number {i} is here.\n\nTranslate one word.",
            )
            for i in range(30)
        ]
        rows.append(row("variant", "linguini", rows[0]["request"].split("\n\n")[0] + "\n\nTranslate another word."))
        mini = [{"input": rows[0]["request"], "target": "unused"}]
        metadata, _, _ = partition(rows, mini)
        by_id = {item["id"]: item for item in metadata}
        self.assertEqual(by_id["0"]["role"], "reserved")
        self.assertEqual(by_id["variant"]["role"], "reserved")
        rows.append(row("conflicting", "linguini", rows[1]["request"], "different"))
        metadata, _, conflicts = partition(rows, mini)
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(all(item["role"] == "excluded" for item in metadata if item["id"] in {"1", "conflicting"}))
        self.assertEqual({item["role"] for item in metadata if item["retained"]}, set(ROLES))

    def test_duplicate_normalization_preserves_semantic_case(self):
        self.assertEqual(duplicate_key("A\n B"), duplicate_key("A  B"))
        self.assertNotEqual(duplicate_key("A B"), duplicate_key("a b"))


if __name__ == "__main__":
    unittest.main()
