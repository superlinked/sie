"""Small synthetic frontier and standalone-HTML safety checks; no model or data access."""

from copy import deepcopy
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import frontier


def point(identity, cost, gap, *, method="embedding", candidate=0, total=None):
    return {
        "point_id": identity,
        "method": method,
        "candidate_id": candidate,
        "rule": {"mode": "threshold", "cutoff": 0.5},
        "answer_accuracy_percent": 94 - gap,
        "quality_gap_pp": gap,
        "answer_usd_per_10000": cost,
        "router_usd_per_10000": None,
        "total_usd_per_10000": total,
        "strong_share": 0.5,
        "analytic_random": method == "random",
    }


class FrontierTests(unittest.TestCase):
    def test_exact_dominance_retains_ties_negative_gaps_and_tradeoffs(self):
        points = [
            point("a", 1, 2),
            point("tie", 1, 2),
            point("worse-gap", 1, 3),
            point("worse-cost", 2, 2),
            point("negative", 3, -1),
            point("better", 4, -2),
            point("almost-tie", 4, -2 + 1e-12),
            point("random", 0, -5, method="random"),
        ]
        self.assertEqual(frontier.pareto_ids(points, "answer_usd_per_10000"), {"a", "tie", "negative", "better"})

    def test_total_cost_does_not_turn_unpriced_hosting_into_zero(self):
        points = [
            point("native", 1, 0, method="lora"),
            point("embedding", 2, 1, total=2.1),
            point("bypass", 3, 0, method="always_strong", candidate=None, total=3),
        ]
        self.assertEqual(frontier.pareto_ids(points, "total_usd_per_10000"), {"embedding", "bypass"})

    def test_script_and_shell_escaping_output_contract_and_no_mutation(self):
        hostile = "</script><script>alert('no')</script>&\u2028"
        report = {"format": "invented", "calibration": [point("candidate", 2, 0)], "provenance": {"note": hostile}}
        report["calibration"][0]["rule"] = hostile
        original = deepcopy(report)
        run = "my run/'; $(touch nope) <script>"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frontier.html"
            receipt = frontier.write_view(report, output, run)
            html = output.read_text()
            self.assertNotIn(hostile, html)
            payload = json.loads(
                html.split('<script id="frontier-data" type="application/json">')[1].split("</script>")[0]
            )
            self.assertEqual(payload["provenance"]["note"], hostile)
            args = shlex.split(payload["calibration"][0]["live_command"].splitlines()[1])
            self.assertEqual(args[args.index("--run") + 1], run)
            self.assertEqual(args[args.index("--candidate-id") + 1], "0")
            self.assertFalse(output.with_suffix(".json").exists())
            json.dumps(receipt, allow_nan=False)
            with self.assertRaises(FileExistsError):
                frontier.write_view(report, output, run)
        self.assertEqual(report, original)

    def test_test_points_never_gain_commands_and_join_is_exact(self):
        cal = point("same", 2, 0, method="lora", candidate=4)
        test = {**cal, "answer_accuracy_percent": 91, "quality_gap_pp": 3}
        payload = frontier._payload({"calibration": [cal], "test": [test]}, "run")
        self.assertIn("--group training", payload["calibration"][0]["live_command"])
        self.assertNotIn("live_command", payload["test"][0])
        self.assertIn('if(state.role!=="calibration")', frontier._HTML)
        with self.assertRaises(ValueError):
            frontier._payload({"calibration": [cal], "test": [{**test, "candidate_id": 5}]}, "run")

    def test_diagnostics_are_excluded_and_malformed_metrics_fail(self):
        p = point("good", 2, 0)
        diagnostic = point("diagnostic", 0, -10, method="metadata")
        payload = frontier._payload({"calibration": [p, diagnostic]}, "run")
        self.assertEqual(len(payload["calibration"]), 1)
        for bad in (float("nan"), float("inf"), True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                frontier._payload({"calibration": [{**p, "quality_gap_pp": bad}]}, "run")
        with self.assertRaises(ValueError):
            frontier._payload({"calibration": [p, p]}, "run")


if __name__ == "__main__":
    unittest.main()
