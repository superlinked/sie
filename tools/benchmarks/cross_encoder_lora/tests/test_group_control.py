"""Dedicated socket ownership, real uvicorn cancellation and bounded teardown."""

# All Python children and signals are scoped to new owned process groups.
# ruff: noqa: S603

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.group_control import Member, decode, envelope
from tools.benchmarks.cross_encoder_lora.interleaved import Peer, cleanup, join_cleanup, watch_peer
from tools.benchmarks.cross_encoder_lora.loadgen import Control
from tools.benchmarks.cross_encoder_lora.runner import ROOT, source_environment
from tools.benchmarks.cross_encoder_lora.tests.test_interleaved import binding
from tools.benchmarks.cross_encoder_lora.tests.test_isolation import sdk_server  # noqa: F401 -- pytest fixture

HELPER = r"""
import asyncio, json, os, signal, socket, sys
from pathlib import Path
from tools.benchmarks.cross_encoder_lora.group_control import Member
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json
value=json.loads(Path(sys.argv[1]).read_text())
owner=Member(int(sys.argv[2]), value, Path(value['output']))
# These are imported only after ownership is active, as in the actual entrypoint.
import uvicorn
from tools.benchmarks.cross_encoder_lora.isolation import IsolatedLoadgen
from tools.benchmarks.cross_encoder_lora.runner import ROOT
async def app(scope, receive, send):
    if scope['type']=='lifespan':
        while True:
            message=await receive()
            await send({'type': message['type']+'.complete'})
            if message['type']=='lifespan.shutdown': return
    else:
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'ok'})
async def run():
    owner.bind_task()
    await owner.grant('prepare')
    sock=socket.socket(); sock.bind(('127.0.0.1',0))
    server=uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    task=asyncio.create_task(server.serve(sockets=[sock]))
    loadgen=None
    try:
        while not server.started: await asyncio.sleep(.01)
        print('ordinary model/log chatter {not protocol}',flush=True)
        async with IsolatedLoadgen(sys.argv[3],ROOT,Evidence(Path(value['output'])/'sdk'),smoke=True,
                                  idle_deadline=value['work_deadline']) as loadgen:
            owner.acknowledge(sdk_pid=loadgen.process.pid,uvicorn_handler=repr(signal.getsignal(signal.SIGTERM)))
            await owner.grant('C1')
            record=await loadgen.phase(1)
            owner.acknowledge(record=record)
            await owner.grant('C4')
    finally:
        server.should_exit=True
        await asyncio.wait_for(task,5)
        sock.close()
        atomic_json(Path(value['output'])/'helper-cleanup.json',{'loadgen':loadgen.receipt if loadgen else None})
try:
    asyncio.run(run())
finally:
    owner.close()
"""


@pytest.mark.parametrize("failure", ["eof", "wrong_identity", "stale_stage", "sdk_crash"])
def test_actual_uvicorn_eof_cancels_and_reaps_sdk_preserving_phase(
    request: pytest.FixtureRequest, tmp_path: Path, failure: str
) -> None:
    url = request.getfixturevalue("sdk_server")
    value = binding(tmp_path)
    value["coordinator_pid"] = os.getpid()
    path = tmp_path / "binding.json"
    atomic_json(path, value)
    parent, child = socket.socketpair()
    process = None
    with (tmp_path / "worker.log").open("wb") as log:
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", HELPER, str(path), str(child.fileno()), url],
                pass_fds=(child.fileno(),),
                close_fds=True,
                start_new_session=True,
                stdout=log,
                stderr=log,
                env=source_environment(ROOT),
            )
            child.close()
            parent.settimeout(10)
            reader = parent.makefile("rb", buffering=0)
            hello = decode(reader.readline(), value)
            assert hello["pid"] == process.pid

            def grant(stage: str, ordinal: int) -> None:
                parent.sendall(
                    (
                        json.dumps(envelope(value, command="grant", stage=stage, ordinal=ordinal, lease="b" * 32))
                        + "\n"
                    ).encode()
                )

            grant("prepare", 0)
            prepared = decode(reader.readline(), value)
            assert (
                "Server.capture_signals" in prepared["uvicorn_handler"]
                or "Server.handle_exit" in prepared["uvicorn_handler"]
            )
            sdk_pid = prepared["sdk_pid"]
            grant("C1", 1)
            complete = decode(reader.readline(), value)
            completed_file = tmp_path / "sdk/loadgen/phase-C1.json"
            original_hash = digest(completed_file)
            assert original_hash == complete["record"]["sha256"]
            if failure == "wrong_identity":
                parent.sendall(b'{"binding_sha256":"wrong"}\n')
            elif failure == "stale_stage":
                grant("C1", 1)
            elif failure == "sdk_crash":
                os.kill(sdk_pid, signal.SIGKILL)
            else:
                reader.close()
                parent.close()
            assert process.wait(timeout=12) != 0
            receipt = json.loads((tmp_path / "helper-cleanup.json").read_text())
            assert receipt["loadgen"]["reaped"]
            assert digest(completed_file) == original_hash
            with pytest.raises(ProcessLookupError):
                os.kill(sdk_pid, 0)
            assert "ordinary model/log chatter" in (tmp_path / "worker.log").read_text()
        finally:
            parent.close()
            child.close()
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_one_shared_cleanup_and_preserves_primary() -> None:
    reaped = []

    async def cleanup():
        await asyncio.sleep(0.08)
        reaped.extend(range(6))
        return "complete"

    primary = ValueError("original phase failure")
    task = asyncio.create_task(cleanup())
    owner = asyncio.create_task(join_cleanup(task, primary))
    await asyncio.sleep(0.01)
    owner.cancel()
    await asyncio.sleep(0.01)
    owner.cancel()
    assert await owner == "complete"
    assert reaped == list(range(6))
    assert task.done()
    assert str(primary) == "original phase failure"
    assert "owned reaping" in primary.__notes__[0]


@pytest.mark.asyncio
async def test_cancel_during_clean_exit_propagates_after_reaping() -> None:
    reaped = []

    async def cleanup():
        await asyncio.sleep(0.03)
        reaped.append(True)

    task = asyncio.create_task(cleanup())
    owner = asyncio.create_task(join_cleanup(task, None))
    await asyncio.sleep(0.01)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert reaped
    assert task.done()


def test_closed_event_loop_cannot_disable_hard_fallback() -> None:
    owner = object.__new__(Member)
    owner.loop = asyncio.new_event_loop()
    owner.loop.close()
    owner.owner = type("Owner", (), {"cancel": lambda *_: None})()
    owner._fail("parent gone")
    assert owner.failure == "parent gone"


def test_group_idle_deadline_is_finite_and_only_for_idle_commands() -> None:
    owner = object.__new__(Control)

    class Messages:
        def get(self, *, timeout):
            assert 30 < timeout <= 60
            return {"token": "owned", "command": "phase", "concurrency": 4}

    owner.messages = Messages()
    owner.token = "owned"  # noqa: S105 -- local control nonce fixture
    assert owner.receive("phase", 4, idle_deadline=time.monotonic() + 60)["concurrency"] == 4
    with pytest.raises(AssertionError, match="active handshake"):
        owner.receive("go", 4, idle_deadline=time.monotonic() + 60)
    with pytest.raises(AssertionError, match="expired"):
        owner.receive("phase", 4, idle_deadline=time.monotonic() - 1)


COORDINATOR = r"""
import json, os, socket, subprocess, sys, time
from pathlib import Path
from tools.benchmarks.cross_encoder_lora.evidence import atomic_json,digest
from tools.benchmarks.cross_encoder_lora.group_control import decode,envelope
from tools.benchmarks.cross_encoder_lora.runner import ROOT,source_environment
root=Path(sys.argv[1]); value=json.loads((root/'template.json').read_text())
owned=[]; sockets=[]; records=[]
for role in 'BNR':
    directory=root/role; directory.mkdir()
    b={**value,'logical_role':role,'output':str(directory),'coordinator_pid':os.getpid()}
    path=directory/'binding.json';atomic_json(path,b)
    parent,child=socket.socketpair();sockets.append(parent)
    log=(directory/'worker.log').open('wb')
    p=subprocess.Popen([sys.executable,str(root/'helper.py'),str(path),str(child.fileno()),sys.argv[2]],
        pass_fds=(child.fileno(),),close_fds=True,start_new_session=True,stdout=log,stderr=log,env=source_environment(ROOT))
    child.close();parent.settimeout(15);reader=parent.makefile('rb',buffering=0)
    decode(reader.readline(),b)
    def grant(stage,ordinal):
        parent.sendall((json.dumps(envelope(b,command='grant',stage=stage,ordinal=ordinal,lease='b'*32))+'\n').encode())
    grant('prepare',0); prepared=decode(reader.readline(),b)
    grant('C1',1); complete=decode(reader.readline(),b)
    records.append({'pid':p.pid,'sdk_pid':prepared['sdk_pid'],'directory':str(directory),
                    'phase_sha256':complete['record']['sha256']})
    owned.append((p,reader,log))
atomic_json(root/'ready.json',{'children':records})
# Parent is deliberately SIGKILLed by the test. No Python finally is expected.
while True: time.sleep(60)
"""


def test_actual_coordinator_sigkill_ends_all_three_uvicorn_sdk_pairs(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    url = request.getfixturevalue("sdk_server")
    atomic_json(tmp_path / "template.json", binding(tmp_path))
    (tmp_path / "helper.py").write_text(HELPER)
    process = subprocess.Popen(
        [sys.executable, "-c", COORDINATOR, str(tmp_path), url],
        env=source_environment(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    rows = []
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.02)
        rows = json.loads((tmp_path / "ready.json").read_text())["children"]
        assert len(rows) == 3
        process.kill()
        assert process.wait(timeout=5) == -signal.SIGKILL
        deadline = time.monotonic() + 10
        while not all((Path(row["directory"]) / "helper-cleanup.json").exists() for row in rows):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        for row in rows:
            directory = Path(row["directory"])
            receipt = json.loads((directory / "helper-cleanup.json").read_text())
            assert receipt["loadgen"]["reaped"]
            assert digest(directory / "sdk/loadgen/phase-C1.json") == row["phase_sha256"]
            with pytest.raises(ProcessLookupError):
                os.kill(row["sdk_pid"], 0)
        deadline = time.monotonic() + 10
        for row in rows:
            while True:
                try:
                    os.kill(row["pid"], 0)
                except ProcessLookupError:
                    break
                assert time.monotonic() < deadline, "server survived coordinator loss"
                time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for row in rows:
            try:
                os.killpg(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_simultaneous_teardown_uses_one_shared_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signalled = []
    gates = [asyncio.Event() for _ in range(3)]

    class Process:
        def __init__(self, index):
            self.pid = index + 1
            self.returncode = None

        async def wait(self):
            await gates[self.pid - 1].wait()
            self.returncode = -signal.SIGKILL
            return self.returncode

    def killpg(pid, sig):
        signalled.append((pid, sig, time.monotonic()))
        if sig == signal.SIGKILL:
            gates[pid - 1].set()

    monkeypatch.setattr(os, "killpg", killpg)
    peers = [Peer({"logical_role": role}, (tmp_path / f"{role}.log").open("wb")) for role in "BNR"]
    for index, peer in enumerate(peers):
        peer.process = cast("asyncio.subprocess.Process", Process(index))
    started = time.monotonic()
    receipt = await cleanup(peers, started + 0.5)
    assert time.monotonic() - started < 0.5
    assert [row[1] for row in signalled[:3]] == [signal.SIGTERM] * 3
    assert [row[1] for row in signalled[3:]] == [signal.SIGKILL] * 3
    assert all(row["reaped"] for row in receipt)


@pytest.mark.asyncio
async def test_inactive_worker_failure_cancels_group() -> None:
    done = asyncio.Event()

    class Process:
        async def wait(self):
            await done.wait()

    peer = object.__new__(Peer)
    peer.process = Process()
    peer.binding = {"logical_role": "R"}
    peer.finishing = peer.stopping = False

    async def active_group():
        owner = asyncio.current_task()
        assert owner is not None
        watcher = asyncio.create_task(watch_peer(peer, owner))
        try:
            await asyncio.sleep(10)
        finally:
            await watcher

    task = asyncio.create_task(active_group())
    await asyncio.sleep(0)
    done.set()
    with pytest.raises(asyncio.CancelledError, match="before group finish"):
        await task
