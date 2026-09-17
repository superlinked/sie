"""Shared configuration for the watermark robustness eval."""

from pathlib import Path

HERE = Path(__file__).parent
SAMPLES_PATH = HERE / "samples.json"
RESULTS_PATH = HERE / "results.json"
REPORT_PATH = HERE / "report.md"

WM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Kirchenbauer-style green-list watermark (transformers built-in).
# These exact values are persisted into samples.json; detection always
# reconstructs the config from the file, never from this module, so the
# two can't drift apart.
WATERMARK_PARAMS = {
    "greenlist_ratio": 0.25,
    "bias": 4.0,
    "hashing_key": 15485863,
    "seeding_scheme": "lefthash",
    "context_width": 1,
}

SIE_URL = "http://localhost:8090"
TRANSLATION_MODEL = "google/madlad400-3b-mt"
NER_MODEL = "urchade/gliner_multi-v2.1"

NER_LABELS = ["person", "organization", "location", "product", "law", "date"]

PROMPTS = [
    "Write a ~150 word opinion column about why cities should invest in night buses.",
    "Write a ~150 word news-style report on a fictional startup called Verdant Loop that recycles coffee grounds into building insulation in Rotterdam.",
    "Write a ~150 word essay on how the printing press changed the economics of rumor.",
    "Write a ~150 word travel piece about visiting Lake Balaton in October.",
    "Write a ~150 word explainer on why container ships keep getting bigger.",
    "Write a ~150 word review of a fictional novel titled 'The Cartographer's Debt' by Ines Marlowe.",
    "Write a ~150 word article about the revival of sleeper trains between Vienna and Brussels.",
    "Write a ~150 word piece on what makes sourdough starters so hard to kill.",
]
