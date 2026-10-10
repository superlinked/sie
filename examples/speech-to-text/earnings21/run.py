"""Fresh full-frame Earnings21 submission through a configured public SIE SDK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import re
import signal
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from sie_sdk import SIEClient

from common import (
    DEFAULT_MODEL,
    OPTIONS,
    SOURCE,
    PersistenceFailure,
    canonical,
    load_frame,
    normalize_url,
    qualify_pieces,
    read_piece,
    redact,
    save_atomic,
    save_new,
    sync_directory,
)
from transport import CaseDeadline, ClosureFailure, OnePost, recover, response_result


def wire_index(wire: list[dict]) -> list[dict]:
    return [
        {
            "file": f"wire-{index:03d}.json",
            **{key: row[key] for key in ("method", "status", "complete", "decoded_sha256")},
        }
        for index, row in enumerate(wire)
    ]


def piece_worker(piece: dict, folder: Path, settings: dict, halt, inner: httpx.BaseTransport | None = None) -> None:
    logging.getLogger("httpx").disabled = True  # Continuation queries can contain opaque credentials.
    started = time.monotonic()
    row = {"status": "known_unattempted", "text": "", "halt": True}
    transport, http, client = None, None, None
    try:
        audio = read_piece(Path(settings["pieces_dir"]), piece)
        transport = OnePost(piece, folder, settings, halt, inner)
        http = httpx.Client(
            base_url=settings["url"],
            transport=transport,
            headers={"Accept-Encoding": "gzip, deflate, identity"},
            follow_redirects=False,
            trust_env=False,
        )
        client = SIEClient(
            settings["url"], api_key=settings["api_key"], timeout_s=transport.remaining(), http_client=http
        )
        reply = client.extract(
            settings["model"],
            {"audio": {"data": audio, "format": "wav"}},
            options=OPTIONS.copy(),
            wait_for_capacity=False,
            max_oom_retries=0,
            provision_timeout_s=transport.remaining(),
        )
        safe_reply = redact(reply, settings["api_key"])
        save_new(folder / "sdk-result.json", canonical(safe_reply))
        row = response_result(transport.wire[-1])
        row["sdk_result_file"] = "sdk-result.json"
        row["sdk_result_known_secret_redacted"] = safe_reply != reply
        data = safe_reply.get("data") if isinstance(safe_reply, dict) else None
        if not isinstance(data, dict) or not isinstance(data.get("text"), str) or safe_reply.get("error") is not None:
            row.update(status="failed", text="", reason="FailedExtractOutput")
        else:
            row.update(status="ok", text=data["text"], data=data)
    except Exception as exc:
        if transport is not None and transport.sends:
            complete = [record for record in transport.wire if record["complete"] and record["status"] != 303]
            row = response_result(complete[-1]) if complete else {"status": "UNKNOWN", "text": "", "halt": True}
            if complete:
                row.update(status="failed", text="")
        row["reason"] = type(exc).__name__  # Never retain exception strings, request headers, or continuation URLs.
        if isinstance(exc, (PersistenceFailure, ClosureFailure, CaseDeadline)) or row["status"] != "failed":
            row["halt"] = True
        if row["halt"]:
            halt.set()
    finally:
        try:
            if client is not None:
                client.close()
            elif http is not None:
                http.close()
        except Exception:
            halt.set()
            row.update(halt=True, closure_failed=True)
        if time.monotonic() >= settings["case_deadline_at"]:
            halt.set()
            row.update(halt=True, deadline_exceeded=True)
    row.update(
        physical_posts=transport.sends if transport is not None else 0,
        wire=wire_index(transport.wire) if transport is not None else [],
        case_start_monotonic=settings["case_start_monotonic"],
        wall_s=time.monotonic() - settings["case_start_monotonic"],
        worker_wall_s=time.monotonic() - started,
        closure_verified=not row.get("closure_failed", False),
    )
    try:
        save_new(folder / "terminal.json", canonical(row))
    except Exception:
        halt.set()
        raise


def child_entry(piece: dict, folder: Path, settings: dict, halt, worker) -> None:
    try:
        if os.name == "posix":
            os.setsid()
        worker(piece, folder, settings, halt)
    except BaseException as exc:
        halt.set()
        try:
            save_new(folder / "worker-failure.json", canonical({"reason": type(exc).__name__, "halt": True}))
        except Exception:
            pass


def call_rows(calls: list[dict], rows: list[dict]) -> list[dict]:
    by_call = {call["id"]: [] for call in calls}
    for row in rows:
        by_call[row["call_id"]].append(row)
    result = []
    for call in calls:
        pieces = sorted(by_call[call["id"]], key=lambda row: row["piece_index"])
        counts = dict(Counter(piece["status"] for piece in pieces))
        error = (
            None
            if all(piece["status"] == "ok" for piece in pieces)
            else ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()) if status != "ok")
        )
        result.append(
            {
                "id": call["id"],
                "hyp": " ".join(piece["text"] if piece["status"] == "ok" else "" for piece in pieces),
                "error": error,
                "complete": all(piece["status"] in {"ok", "failed"} for piece in pieces),
                "piece_ids": [piece["id"] for piece in pieces],
                "piece_status_counts": counts,
            }
        )
    return result


def checkpoint(output: Path, calls: list[dict], rows: list[dict]) -> None:
    aggregates = call_rows(calls, rows)
    save_atomic(output / "checkpoint.json", canonical({"pieces": rows, "calls": aggregates}))
    save_atomic(output / "pieces.jsonl", b"".join(canonical(row) + b"\n" for row in rows))
    save_atomic(output / "calls.jsonl", b"".join(canonical(row) + b"\n" for row in aggregates))
    save_atomic(
        output / "hypotheses.jsonl",
        b"".join(canonical({key: row[key] for key in ("id", "hyp", "error")}) + b"\n" for row in aggregates),
    )


def signal_owned(process, number: int) -> None:
    if os.name == "posix" and process.pid is not None:
        try:
            os.killpg(process.pid, number)
            return
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill() if number == signal.SIGKILL else process.terminate()


def execute(
    calls: list[dict], pieces: list[dict], output: Path, settings: dict, *, worker=piece_worker, context=None
) -> list[dict]:
    """One parent owns dispatch, halt state, aggregate checkpoints, and bounded reaping."""
    validate_settings(settings)
    started = time.monotonic()
    deadline = started + settings["whole_seconds"]
    reserve = min(2.0, settings["whole_seconds"] / 5, settings["case_seconds"] / 5)
    work_deadline = deadline - reserve
    context = context or multiprocessing.get_context("spawn")
    halt = context.Event()
    active = {}
    owned = []
    rows = [
        {**piece, "status": "known_unattempted", "text": "", "halt": False, "physical_posts": 0} for piece in pieces
    ]
    output.mkdir(exist_ok=False)
    sync_directory(output.parent)
    cases = output / "pieces"
    cases.mkdir()
    sync_directory(output)
    for piece in pieces:
        (cases / piece["id"]).mkdir()
    sync_directory(cases)
    plan = {key: value for key, value in settings.items() if key not in {"api_key", "pieces_dir"}}
    plan.update(
        source_revision=SOURCE["revision"],
        source_manifest_sha256=SOURCE["files"]["manifest.json"]["sha256"],
        requested_model=settings["model"],
        requested_arm="S3" if settings["model"] == DEFAULT_MODEL else "model-override",
        expected_source_model_revision=SOURCE["expected_model_revision"]
        if settings["model"] == DEFAULT_MODEL
        else None,
        server_reported_revision=None,
        utc_start=datetime.now(UTC).isoformat(),
        monotonic_start=started,
        whole_deadline_at=deadline,
        shutdown_reserve_s=reserve,
        calls=len(calls),
        pieces=len(pieces),
        sdk_version=version("sie-sdk"),
        options=OPTIONS,
        one_physical_post_per_piece=True,
    )
    safe_plan = redact(plan, settings["api_key"])
    safe_plan["known_secret_redacted"] = safe_plan != plan
    save_new(output / "plan.json", canonical(safe_plan))
    checkpoint(output, calls, rows)
    next_index = 0
    failure = None
    try:
        while active or (next_index < len(pieces) and not halt.is_set()):
            now = time.monotonic()
            if now >= work_deadline:
                halt.set()
            for index, state in list(active.items()):
                process, case_deadline, terminated = state
                if process.is_alive() and now >= case_deadline:
                    halt.set()
                    if terminated is None:
                        signal_owned(process, signal.SIGTERM)
                        state[2] = now
                    elif now >= min(terminated + 0.1, deadline - reserve / 2):
                        signal_owned(process, signal.SIGKILL)
                if not process.is_alive():
                    process.join(timeout=0)
                    signal_owned(process, signal.SIGKILL)  # Remove any descendants in this owned process group.
                    terminal = recover(cases / pieces[index]["id"])
                    terminal["wall_s"] = time.monotonic() - rows[index]["case_start_monotonic"]
                    if terminated is not None:
                        terminal.update(halt=True, deadline_exceeded=True, termination_reason="WallDeadline")
                    terminal["wire"] = (
                        wire_index(terminal["wire"])
                        if terminal.get("reconstructed_from_response")
                        else terminal.get("wire", [])
                    )
                    rows[index].update(terminal)
                    if terminal.get("halt") or process.exitcode != 0:
                        halt.set()
                    del active[index]
                    checkpoint(output, calls, rows)
            # UNKNOWN/persistence/closure signals are checked before every replacement, including initial dispatch.
            while not halt.is_set() and next_index < len(pieces) and len(active) < settings["concurrency"]:
                now = time.monotonic()
                if now >= work_deadline:
                    halt.set()
                    break
                index = next_index
                case_settings = {
                    **settings,
                    "case_start_monotonic": now,
                    "case_deadline_at": min(now + settings["case_seconds"], work_deadline),
                }
                process = context.Process(
                    target=child_entry, args=(pieces[index], cases / pieces[index]["id"], case_settings, halt, worker)
                )
                owned.append(process)
                process.start()
                rows[index].update(status="running", case_start_monotonic=now, worker_pid=process.pid)
                active[index] = [process, case_settings["case_deadline_at"], None]
                next_index += 1
            if active:
                time.sleep(min(0.01, max(0, work_deadline - time.monotonic())))
    except BaseException as exc:
        failure = exc
        halt.set()
    finally:
        # One shared grace interval; it never multiplies with the number of children.
        for process in owned:
            if process.pid is not None and process.is_alive():
                signal_owned(process, signal.SIGTERM)
        grace = min(time.monotonic() + reserve / 2, deadline - reserve / 2)
        while any(process.pid is not None and process.is_alive() for process in owned) and time.monotonic() < grace:
            time.sleep(0.005)
        for process in owned:
            if process.pid is not None:
                signal_owned(process, signal.SIGKILL)
        for process in owned:
            if process.pid is not None:
                process.join(timeout=max(0, deadline - time.monotonic()))
        if any(process.pid is not None and process.is_alive() for process in owned):
            raise RuntimeError("Owned worker did not exit after kill; run failed")
        for index in active:
            rows[index].update(recover(cases / pieces[index]["id"]))
        try:
            checkpoint(output, calls, rows)
            save_new(
                output / "finished.json",
                canonical(
                    {"halted": halt.is_set(), "wall_s": time.monotonic() - started, "owned_children_reaped": True}
                ),
            )
        except Exception as exc:
            failure = failure or exc
        for process in owned:
            if process.pid is not None:
                process.close()
    if failure is not None:
        raise failure
    return rows


def validate_settings(settings: dict) -> None:
    if version("sie-sdk") != "0.10.0":
        raise ValueError("The released sie-sdk 0.10.0 is required")
    normalize_url(settings["url"])
    for key in ("case_seconds", "whole_seconds"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Deadlines must be finite positive seconds")
    if type(settings["concurrency"]) is not int or not 1 <= settings["concurrency"] <= 8:
        raise ValueError("Concurrency must be between 1 and 8")
    if not re.fullmatch(r"[A-Za-z0-9_./:-]{1,512}", settings["model"]):
        raise ValueError("Model must be a nonempty SIE model identifier")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    parser.add_argument("--pieces-dir", type=Path, default=Path("data/pieces"))
    parser.add_argument("--output", type=Path, help="New output directory; an existing directory is never executable")
    parser.add_argument("--url", default=os.environ.get("SIE_URL"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--case-seconds", type=float, default=1200)
    parser.add_argument("--whole-seconds", type=float, default=21600)
    parser.add_argument(
        "--verify-only", action="store_true", help="Check the complete 44/219 input frame without sending"
    )
    args = parser.parse_args()
    try:
        calls, pieces = load_frame(args.evidence)
        qualify_pieces(args.pieces_dir, pieces)
        if args.verify_only:
            print("Verified the original 44-call / 219-piece frame; no requests sent.")
            return 0
        if args.url is None or args.output is None:
            raise ValueError("Set SIE_URL or --url, and select a new --output directory")
        settings = {
            "url": normalize_url(args.url),
            "model": args.model,
            "api_key": os.environ.get("SIE_API_KEY", ""),
            "pieces_dir": str(args.pieces_dir.resolve()),
            "concurrency": args.concurrency,
            "case_seconds": args.case_seconds,
            "whole_seconds": args.whole_seconds,
        }
        rows = execute(calls, pieces, args.output, settings)
        counts = dict(Counter(row["status"] for row in rows))
        print(f"Retained all 219 pieces and 44 calls: {counts}. Output: {args.output}")
        halted = json.loads((args.output / "finished.json").read_bytes())["halted"]
        return 0 if not halted and all(row["status"] == "ok" for row in rows) else 1
    except Exception as exc:
        print(f"Caller stopped: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
