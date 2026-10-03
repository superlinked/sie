"""Verify a committed run offline, without calling the API.

Why this exists
---------------
``evaluate.py`` scores ``<run>/markdown/<slug>.md`` and never opens the
recorded response beside it. That gap was found by running this example: the
Markdown file the checks read is written by the harness itself, so editing that
file alone would make all 25 checks pass while the recorded API response said
something else. The checks would be confirming the artifact under test.

This module closes the gap without changing what the checks score. It asserts
the *relation* between the scored file and the response:

    markdown_file_text == response.data.markdown.rstrip() + "\\n"

exactly, rather than normalizing both sides and comparing. A lenient comparison
would still pass if ``convert.py`` later changed its transform, which is the
one thing this check exists to catch.

Everything else here is digest arithmetic over canonical content, so it runs
with no network and no key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from document_to_markdown.canonical import canonical_sha256
from document_to_markdown.config import PDF_DIR, ROOT, is_fetched_bundle

console = Console()


class VerificationError(Exception):
    """A recorded digest or relation did not hold."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        # A file the verifier needs and cannot find is a failure, never a skip.
        # Reporting it as one check that could not run would leave the total
        # reading as a pass.
        raise VerificationError(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _check(results: list[tuple[bool, str]], ok: bool, message: str) -> None:
    results.append((ok, message))


def _relative(path: Path) -> str:
    """The path as the README names it, for a check message."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sources_manifest_for(run_dir: Path) -> Path:
    """Where the provenance for THIS run's source PDFs lives.

    A bundle fetched from the dataset carries its own under
    ``inputs/sources.json``, describing the PDFs that run actually read. A
    bundle produced locally by ``convert-documents`` carries none, so the
    manifest ``fetch-documents`` wrote beside the PDFs is the one that
    describes what it read.

    A fixed path would check a local run against somebody else's provenance,
    which is the same shape as the duplicate-slug problem below: two checks
    reading rows that were never about the same file.

    Which bundle this is decides the answer, never whether the file happens to
    be there. A fetched bundle missing its ``inputs/sources.json`` gets that
    path back regardless, so the presence check fails; falling back would let
    whatever PDFs are sitting in ``pdfs/`` satisfy the provenance checks for a
    different set of artifacts, which is the failure this join exists to
    prevent. A local bundle with no PDF manifest fails the same way.
    """
    bundled = run_dir / "inputs" / "sources.json"
    return bundled if is_fetched_bundle(run_dir) else PDF_DIR / "manifest.json"


def _resolve_payloads(value: Any, run_dir: Path, results: list[tuple[bool, str]]) -> Any:
    """Replace every ``$payload`` reference with its stored value, verifying it."""
    if isinstance(value, dict):
        reference = value.get("$payload")
        if isinstance(reference, dict) and set(reference) >= {"sha256", "path"}:
            # The path comes out of the artifact being verified, so it is not
            # trusted input. An absolute path, a "..", or a symlink would make
            # the verifier read a file outside the recorded run and then report
            # success over it -- offline integrity that quietly depends on
            # something offline. record.py writes exactly payloads/<digest>.json,
            # so require that and nothing else.
            digest = reference["sha256"]
            expected = f"payloads/{digest}.json"
            if reference["path"] != expected:
                raise VerificationError(
                    f"payload reference {reference['path']!r} is not the digest-addressed {expected!r}"
                )
            payloads_root = (run_dir / "payloads").resolve()
            payload_path = (run_dir / expected).resolve()
            if payload_path.parent != payloads_root:
                raise VerificationError(f"payload {reference['path']} resolves outside {payloads_root}")
            if not payload_path.exists():
                raise VerificationError(f"missing payload {reference['path']}")
            # Digested over stored bytes, never parsed and re-serialized: these
            # subtrees can hold integers outside the IEEE 754 exact range.
            # `shasum -a 256` on the file reproduces this value.
            body = payload_path.read_bytes()
            label = Path(reference["path"]).name[:16]
            _check(results, hashlib.sha256(body).hexdigest() == reference["sha256"], f"payload {label} digest")
            _check(results, len(body) == reference["bytes"], f"payload {label} size")
            return _read_json(payload_path)
        return {key: _resolve_payloads(item, run_dir, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_payloads(item, run_dir, results) for item in value]
    return value


def verify_run(run_dir: Path) -> tuple[list[tuple[bool, str]], list[str]]:
    """Return (check results, slugs whose PDF digest check could not run)."""
    results: list[tuple[bool, str]] = []
    not_checked: list[str] = []
    manifest = _read_json(run_dir / "run-manifest.json")
    calls_document = _read_json(run_dir / "calls.json")

    recorded_manifest_digest = manifest.pop("manifest_sha256", None)
    _check(results, canonical_sha256(manifest) == recorded_manifest_digest, "manifest self-digest")
    manifest["manifest_sha256"] = recorded_manifest_digest

    _check(
        results,
        canonical_sha256(calls_document) == manifest["calls"]["sha256"],
        "calls.json digest matches the manifest",
    )
    _check(
        results,
        len(calls_document["calls"]) == manifest["calls"]["call_count"],
        "recorded call count",
    )

    # A count is not a set. Replacing one call with a duplicate of another keeps
    # call_count intact, verifies the duplicate twice, and leaves the dropped
    # document's Markdown and evaluation bound to nothing -- every check green.
    # So require one-to-one slug coverage across every artifact that names them.
    call_slugs = [call["slug"] for call in calls_document["calls"]]
    _check(results, len(call_slugs) == len(set(call_slugs)), "no call slug appears twice")

    # Every list of rows keyed by slug needs this, because a set conversion
    # hides a duplicate and different lookups then pick different rows from it:
    # `next(...)` takes the first match, a dict comprehension keeps the last. Two
    # rows with the same slug and different digests would satisfy one comparison
    # against one row and the other against the other, both green. So check
    # uniqueness first and then read every row through ONE mapping.
    manifest_slugs = [row["slug"] for row in manifest["documents"]]
    _check(
        results,
        len(manifest_slugs) == len(set(manifest_slugs)),
        "no run-manifest document slug appears twice",
    )
    recorded_documents = {row["slug"]: row for row in manifest["documents"]}

    for label, other in (
        ("manifest.calls.entries", set(manifest["calls"]["entries"])),
        ("manifest.documents", set(recorded_documents)),
    ):
        _check(results, set(call_slugs) == other, f"calls.json and {label} name the same documents")

    for entry in calls_document["calls"]:
        slug = entry["slug"]
        recorded_entry_digest = entry.pop("entry_sha256", None)
        _check(results, canonical_sha256(entry) == recorded_entry_digest, f"{slug}: entry self-digest")
        entry["entry_sha256"] = recorded_entry_digest

        _check(
            results,
            manifest["calls"]["entries"].get(slug) == recorded_entry_digest,
            f"{slug}: manifest lists this entry digest",
        )

        response = _resolve_payloads(entry["response"], run_dir, results)
        markdown = response["data"]["markdown"]

        scored = entry["scored_markdown"]
        # Same untrusted-path problem as the payload reference, in the sibling
        # field. I fixed one of the two last round and did not look for the
        # other: `scored["file"]` also comes out of the artifact being verified,
        # so a "..", an absolute path or a symlink would let a bundle pass the
        # digest and relation checks while containing no scored Markdown at all.
        expected_markdown = f"markdown/{slug}.md"
        if scored["file"] != expected_markdown:
            raise VerificationError(f"scored Markdown reference {scored['file']!r} is not {expected_markdown!r}")
        markdown_root = (run_dir / "markdown").resolve()
        markdown_path = (run_dir / expected_markdown).resolve()
        if markdown_path.parent != markdown_root:
            raise VerificationError(f"scored Markdown {scored['file']} resolves outside {markdown_root}")
        if not markdown_path.exists():
            raise VerificationError(f"missing scored Markdown {scored['file']}")
        file_bytes = markdown_path.read_bytes()
        _check(
            results,
            hashlib.sha256(file_bytes).hexdigest() == scored["sha256"],
            f"{slug}: scored Markdown file digest",
        )
        _check(
            results,
            canonical_sha256(markdown) == scored["response_markdown_sha256"],
            f"{slug}: response Markdown digest",
        )
        # The relation, asserted strictly. See the module docstring.
        _check(
            results,
            file_bytes.decode("utf-8") == markdown.rstrip() + "\n",
            f"{slug}: scored Markdown is exactly the response Markdown under the recorded transform",
        )

        recorded_source = recorded_documents.get(slug)
        # A missing row used to raise StopIteration, which reads as a crash
        # rather than as a verification failure. Record it as a failed check.
        _check(results, recorded_source is not None, f"{slug}: the run manifest lists this document")
        if recorded_source is not None:
            _check(
                results,
                recorded_source["source_sha256"] == entry["request"]["item"]["document_sha256"],
                f"{slug}: source digest agrees between manifest and call",
            )

    evaluation_path = run_dir / "evaluation.json"
    # Required, not optional. Skipping it silently would let an incomplete
    # bundle report success, which is the failure this module exists to prevent.
    _check(results, evaluation_path.exists(), "evaluation.json is present")
    if evaluation_path.exists():
        evaluation = _read_json(evaluation_path)
        for document in evaluation["documents"]:
            counted = sum(1 for check in document["checks"] if check["passed"])
            _check(
                results,
                counted == document["passed"] and len(document["checks"]) == document["total"],
                f"{document['slug']}: totals recomputed from the checks",
            )
        evaluated = [row["slug"] for row in evaluation["documents"]]
        _check(results, len(evaluated) == len(set(evaluated)), "no evaluated slug appears twice")
        _check(
            results,
            set(evaluated) == set(call_slugs),
            "evaluation.json and calls.json name the same documents",
        )
        total = sum(row["total"] for row in evaluation["documents"])
        passed = sum(row["passed"] for row in evaluation["documents"])
        # Whether the run passed is checked for CONSISTENCY, not required to be
        # true. A verifier that refuses to verify a failing run would pressure
        # whoever records the next one into not recording a failure at all.
        # What must hold is that the recorded verdict matches the checks.
        _check(
            results,
            bool(evaluation.get("passed")) == (passed == total),
            f"evaluation.passed={evaluation.get('passed')} agrees with {passed} of {total} checks passing",
        )
        console.print(f"[bold]Recomputed from the check arrays: {passed} of {total} checks passed[/]")

    sources_path = sources_manifest_for(run_dir)
    sources_label = _relative(sources_path)
    # Required, and every scored document must appear in it. Treating either as
    # optional let a missing file or a missing row skip the provenance check
    # while the run still reported success -- the third time that shape has
    # appeared in this module, and the second time I wrote it myself.
    _check(results, sources_path.exists(), f"{sources_label} is present")
    if sources_path.exists():
        sources = _read_json(sources_path)
        # Fourth member of the same class, and the finding did not name it:
        # The sources manifest is a slug-keyed row list too. It has no first/last
        # asymmetry today because nothing builds a dict from it, but tolerating a
        # duplicate here is how that asymmetry arrives later without anyone
        # noticing. Enumerating the class is the point, not patching the
        # instance that was reported.
        source_slugs = [row["slug"] for row in sources["documents"]]
        _check(
            results,
            len(source_slugs) == len(set(source_slugs)),
            f"no {sources_label} slug appears twice",
        )
        listed = set(source_slugs)
        for row in manifest["documents"]:
            _check(
                results,
                row["slug"] in listed,
                f"{row['slug']}: the scored document appears in {sources_label}",
            )
        skipped = []
        # Join the two halves of the provenance chain. Without this, one check
        # says the fetched PDF matches the sources manifest and another says the
        # run scored a document with some digest, and nothing says those are the
        # same file -- so a reader could verify a download that was never the
        # thing this run read.
        scored = {slug: row.get("source_sha256") for slug, row in recorded_documents.items()}
        for row in sources["documents"]:
            if row["slug"] in scored:
                _check(
                    results,
                    row["sha256"] == scored[row["slug"]],
                    f"{row['slug']}: the source this manifest pins is the one the run scored",
                )
            # PDF_DIR rather than a second path built by hand: if the two ever
            # disagreed, every file would look absent, every PDF check would be
            # skipped, and the run would still report success.
            #
            # And `file_name` is untrusted for the same reason the payload and
            # scored-Markdown paths are, with one aggravation: the sources manifest
            # sits OUTSIDE the run bundle, so no recorded digest covers it. Edit
            # the name, the digest and the size together and the source check
            # passes against a file that was never fetched. Third member of this
            # class in this module, after $payload.path and scored_markdown.file.
            relative = Path(row["file_name"])
            pdf_root = PDF_DIR.resolve()
            local = (pdf_root / relative).resolve()
            if relative.name != row["file_name"] or local.parent != pdf_root:
                raise VerificationError(f"source PDF {row['file_name']!r} resolves outside {pdf_root}")
            if not local.exists():
                skipped.append(row["slug"])
                continue
            payload = local.read_bytes()
            _check(
                results,
                hashlib.sha256(payload).hexdigest() == row["sha256"] and len(payload) == row["bytes"],
                f"{row['slug']}: fetched PDF matches the recorded digest",
            )
        # Reported by the caller rather than dropped. The PDFs are fetch-only,
        # so a fresh clone has none of them, and an unqualified "43 of 43
        # passed" would then be true and misleading at once.
        not_checked.extend(skipped)

    return results, not_checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a recorded run without calling the API")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        results, not_checked = verify_run(args.run_dir)
    except VerificationError as error:
        # Exits nonzero with the reason, rather than a traceback that reads
        # like a bug in the verifier instead of a problem with the bundle.
        console.print(f"[red]FAILED[/] {error}")
        sys.exit(1)
    failures = [message for ok, message in results if not ok]
    for ok, message in results:
        console.print(f"[green]  ok  [/] {message}" if ok else f"[red]FAILED[/] {message}")
    for slug in not_checked:
        console.print(f"[yellow] skip [/] {slug}: source PDF not present, so its digest was not checked")
    summary = f"\n{len(results) - len(failures)} of {len(results)} checks passed"
    if not_checked:
        summary += f", {len(not_checked)} not checked — run `uv run fetch-documents` to verify the source PDFs too"
    console.print(summary)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
