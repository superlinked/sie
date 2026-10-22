"""Independent, answer-model-free tests of the source-derived answer contract.

Run with the reviewed collection directory on PYTHONPATH. No recorded answer,
reference label, model identity, or correctness result is loaded by these tests.
"""

import unittest

from normalize import canonicalize


YES_NO = (
    "Your answer should be a list of three words, yes or no "
    "(for example, yes, no, yes)."
)
THREE_WAY = (
    "Your answer should be a list of three words separated by commas. "
    "Each of them must be either yes (if the person tells the truth), "
    "no (if the person lies), or unknown (if the person's truth cannot be "
    "determined). For example, <unknown, no, yes> or <yes, yes, yes>."
)


class SourceCanonicalizationTests(unittest.TestCase):
    def test_both_choice_meanings_survive_without_a_reference(self):
        prompt = "Pick one.\n(A) orange ball\n(B) purple ball"
        for task in (
            "boolean_expressions", "disambiguation_qa", "geometric_shapes",
            "movie_recommendation", "nycc", "shuffled_objects",
        ):
            for label, option in (("A", "orange ball"), ("B", "purple ball")):
                for answer in (label, f"({label})", f"({label}) {option}", option):
                    with self.subTest(task=task, answer=answer):
                        self.assertEqual(canonicalize(answer, prompt, task)[0], f"({label})")

    def test_label_text_contradictions_and_explanations_are_not_extracted(self):
        prompt = "Pick one.\n(A) orange ball\n(B) purple ball"
        for answer in (
            "(A) purple ball", "(B) orange ball", "(A) or (B)", "AB",
            "(A) orange ball; alternatively (B) purple ball", "orange ball because A wins",
            "(A) orange ball\nActually, choose B.", "The answer is (A)",
            "(A) orange ball\nThe final answer is (B)", "(Z) orange ball",
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(canonicalize(answer, prompt, "nycc")[0])

    def test_whole_option_matching_preserves_negation_and_internal_punctuation(self):
        prompt = "Pick one.\n(A) We aren't leaving.\n(B) We are leaving."
        self.assertEqual(canonicalize("(A) We aren't leaving", prompt, "nycc")[0], "(A)")
        for answer in ("(A) We are leaving.", "We arent leaving.", "We are not leaving."):
            with self.subTest(answer=answer):
                self.assertIsNone(canonicalize(answer, prompt, "nycc")[0])

    def test_repeated_text_under_different_labels_is_ambiguous(self):
        prompt = "Pick one.\n(A) same meaning\n(B) different meaning\n(C) same meaning"
        self.assertIsNone(canonicalize("same meaning", prompt, "disambiguation_qa")[0])
        self.assertEqual(canonicalize("(C) same meaning", prompt, "disambiguation_qa")[0], "(C)")

    def test_duplicate_label_retains_upstream_label_semantics_only(self):
        prompt = "Pick one.\n(A) first meaning\n(B) second meaning\n(B) third meaning"
        for answer in ("B", "(B) second meaning", "(B) third meaning", "third meaning"):
            with self.subTest(answer=answer):
                self.assertEqual(canonicalize(answer, prompt, "disambiguation_qa")[0], "(B)")
        self.assertIsNone(canonicalize("(A) third meaning", prompt, "disambiguation_qa")[0])

    def test_geometry_footer_is_not_an_option_or_acceptable_explanation(self):
        footer = "Coordinates have been rounded to 5 decimal places so ignore slight differences."
        prompt = f"Options:\n(A) only 1\n(B) 2 and 3\n{footer}"
        self.assertEqual(canonicalize("(B) 2 and 3", prompt, "geometric_shapes")[0], "(B)")
        self.assertIsNone(canonicalize("(B) 2 and 3 " + footer, prompt, "geometric_shapes")[0])

    def test_truth_tuples_preserve_all_three_positions(self):
        for prompt, answer, expected in (
            (YES_NO, "<yes, no, yes>", "yes, no, yes"),
            (YES_NO, "no, yes, no", "no, yes, no"),
            (THREE_WAY, "<unknown, no, yes>", "unknown, no, yes"),
            (THREE_WAY, "<YES, unknown, NO>", "yes, unknown, no"),
        ):
            with self.subTest(answer=answer):
                value = canonicalize(answer, prompt, "web_of_lies")[0]
                self.assertEqual(value.casefold().replace(" ", ""), expected.replace(" ", ""))

    def test_truth_vocabulary_is_taken_from_the_question(self):
        for answer in ("<unknown, no, yes>", "unknown, no, yes"):
            with self.subTest(answer=answer):
                self.assertIsNone(canonicalize(answer, YES_NO, "web_of_lies")[0])
                self.assertIsNotNone(canonicalize(answer, THREE_WAY, "web_of_lies")[0])

    def test_malformed_truth_tuples_cannot_reach_permissive_upstream_extraction(self):
        for answer in (
            "<yes, no>", "<yes, no, yes, no>", "<<yes, no, yes>>", "<yes, maybe, no>",
            "yes, no, yes\nActually, no, no, no", "<yes, no, yes> because the facts imply it",
            "<yes, no, yes>\nThe final answer is no, no, no", "yes,no,yes or no,no,no",
            "yes, no, yes>", "<yes, no, yes",
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(canonicalize(answer, THREE_WAY, "web_of_lies")[0])

    def test_no_global_list_multiselect_language_or_number_rewriting(self):
        for task, answer in (
            ("hyperbaton", "ABH"), ("sportqa", "AC, D, BD"),
            ("sarc_triples", "0,1,0"), ("linguini", "[ətkəbox]"),
            ("linguini", "l'home"), ("dyck_languages", "<[]>"),
            ("time_arithmetic", "[Emma, Lily]"), ("temporal_sequence", "45, 3"),
            ("multistep_arithmetic", "9007199254740993"), ("boardgame_qa", "disproved"),
            ("causal_understanding", "Ambiguous"), ("object_properties", "unknown"),
            ("spatial_reasoning", "Granny Smith"),
        ):
            with self.subTest(task=task, answer=answer):
                self.assertEqual(canonicalize(answer, "source request", task), (answer, "unchanged"))


if __name__ == "__main__":
    unittest.main()
