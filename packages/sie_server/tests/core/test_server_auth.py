"""Token authentication for sie-server: resolution, the persisted token, and the middleware."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import logging
import os
import re
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core import server_auth
from sie_server.core.data_dir import owned_files, resolve_data_dir
from sie_server.core.server_auth import (
    AuthConfigError,
    LoopbackHostMiddleware,
    ServerAuthMiddleware,
    TokenFileError,
    configured_tokens,
    is_loopback_host,
    load_or_create_token,
    loopback_host_allowlist,
    read_stored_token,
    request_credentials,
    request_host,
    resolve_auth_mode,
    server_tokens,
    token_path,
)
from sie_server.observability.telemetry import _get_or_create_worker_id
from starlette.testclient import WebSocketDenialResponse

TOKEN = "s3cret-token-value"  # noqa: S105 - test credential
OTHER = "another-token-value"
_MACOS_ACL = pytest.mark.skipif(sys.platform != "darwin", reason="macOS access control lists, set with chmod +a")


@pytest.fixture
def add_acl_entry(tmp_path: Path) -> Iterator[Callable[[Path, str], None]]:
    def add(path: Path, entry: str) -> None:
        subprocess.run(["/bin/chmod", "+a", entry, str(path)], check=True)  # noqa: S603

    yield add
    subprocess.run(["/bin/chmod", "-RN", str(tmp_path)], check=True)  # noqa: S603


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = dict(os.environ)
    for name in (
        "SIE_AUTH_MODE",
        "SIE_AUTH_TOKEN",
        "SIE_AUTH_TOKENS",
        "SIE_DATA_DIR",
        "SIE_SERVER_HOST",
        "XDG_DATA_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


class TestResolveAuthMode:
    @pytest.mark.parametrize(
        ("requested", "has_tokens", "exposed", "expected"),
        [
            (None, False, False, "none"),
            (None, False, True, "none"),
            (None, True, False, "token"),
            (None, True, True, "token"),
            ("token", False, False, "token"),
            ("static", False, True, "token"),
            ("auto", False, False, "none"),
            ("auto", False, True, "token"),
            ("auto", True, False, "token"),
            ("none", False, False, "none"),
            ("none", False, True, "none"),
        ],
    )
    def test_resolution(self, requested: str | None, has_tokens: bool, exposed: bool, expected: str) -> None:
        assert resolve_auth_mode(requested, has_tokens=has_tokens, exposed=exposed) == expected

    def test_a_token_with_auth_explicitly_off_is_refused(self) -> None:
        with pytest.raises(AuthConfigError, match="SIE_AUTH_MODE=none"):
            resolve_auth_mode("none", has_tokens=True, exposed=False)

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(AuthConfigError, match="none, token, static or auto"):
            resolve_auth_mode("open", has_tokens=False, exposed=False)

    @pytest.mark.parametrize("requested", ["TOKEN", "Token", "None", "AUTO"])
    def test_modes_are_case_sensitive_as_on_the_gateway(self, requested: str) -> None:
        with pytest.raises(AuthConfigError, match="lowercase"):
            resolve_auth_mode(requested, has_tokens=False, exposed=False)

    @pytest.mark.parametrize("requested", [None, "token", "static", "auto"])
    def test_a_declared_but_blank_token_is_refused(self, requested: str | None) -> None:
        with pytest.raises(AuthConfigError, match="holds no token"):
            resolve_auth_mode(requested, has_tokens=False, exposed=True, declares_tokens=True)

    def test_a_blank_token_with_auth_explicitly_off_is_harmless(self) -> None:
        assert resolve_auth_mode("none", has_tokens=False, exposed=True, declares_tokens=True) == "none"


class TestConfiguredTokens:
    def test_tokens_variable_wins_over_token(self) -> None:
        env = {"SIE_AUTH_TOKENS": " a , b,,", "SIE_AUTH_TOKEN": "c"}
        assert configured_tokens(env) == ("a", "b")

    def test_single_token_variable_is_comma_separated_too(self) -> None:
        assert configured_tokens({"SIE_AUTH_TOKEN": "c,d"}) == ("c", "d")

    def test_blank_values_configure_nothing(self) -> None:
        assert configured_tokens({"SIE_AUTH_TOKENS": " , ", "SIE_AUTH_TOKEN": ""}) == ()


class TestIsLoopbackHost:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "::1", "[::1]", "localhost", "LOCALHOST"])
    def test_loopback(self, host: str) -> None:
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.1", "example.com", ""])  # noqa: S104
    def test_exposed(self, host: str) -> None:
        assert not is_loopback_host(host)


class TestDataDir:
    def test_explicit_directory_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        assert resolve_data_dir() == tmp_path / "data"

    def test_xdg_data_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / "sie"

    def test_relative_xdg_data_home_is_ignored(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", "relative")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / ".local" / "share" / "sie"

    def test_default_is_per_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / ".local" / "share" / "sie"


class TestPersistedToken:
    def test_first_use_creates_a_private_token(self, tmp_path: Path) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)

        assert stored.created
        assert stored.path == token_path(directory)
        assert len(stored.token) >= 43
        assert stat.S_IMODE(stored.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
        assert stored.path.read_text() == f"{stored.token}\n"

    def test_later_use_reads_the_same_token(self, tmp_path: Path) -> None:
        first = load_or_create_token(tmp_path)
        second = load_or_create_token(tmp_path)

        assert not second.created
        assert second.token == first.token

    def test_a_token_others_can_read_is_refused(self, tmp_path: Path) -> None:
        directory = tmp_path / "Application Support"
        stored = load_or_create_token(directory)
        stored.path.chmod(0o644)

        with pytest.raises(TokenFileError, match=re.escape(f"(chmod 600 '{stored.path}')")):
            read_stored_token(directory)

    def test_concurrent_first_starts_agree_on_one_token(self, tmp_path: Path) -> None:
        for trial in range(10):
            directory = tmp_path / f"trial-{trial}"
            starts = [
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-c", _FIRST_START_SCRIPT, str(directory)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(6)
            ]
            results = [start.communicate(timeout=60) for start in starts]
            assert [start.returncode for start in starts] == [0] * 6, results
            assert len({out.strip() for out, _ in results}) == 1
            assert [entry.name for entry in directory.iterdir()] == ["auth-token"]
            assert token_path(directory).stat().st_nlink == 1

    def test_an_empty_token_file_names_the_file_to_delete(self, tmp_path: Path) -> None:
        path = token_path(tmp_path)
        path.touch(mode=0o600)

        with pytest.raises(TokenFileError, match=f"Delete {re.escape(str(path))}"):
            load_or_create_token(tmp_path)

    def test_a_fifo_is_refused_without_blocking(self, tmp_path: Path) -> None:
        os.mkfifo(token_path(tmp_path), 0o600)
        outcome: list[BaseException] = []
        reader = threading.Thread(target=_capture, args=(lambda: read_stored_token(tmp_path), outcome), daemon=True)
        reader.start()
        reader.join(timeout=5)

        assert not reader.is_alive(), "reading a FIFO token path blocked"
        assert isinstance(outcome[0], TokenFileError)
        assert "must be a regular file" in str(outcome[0])

    def test_a_temporary_link_left_by_an_interrupted_start_is_dropped(self, tmp_path: Path) -> None:
        stored = load_or_create_token(tmp_path)
        staged = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        os.link(stored.path, staged)

        assert read_stored_token(tmp_path) == stored.token
        assert not staged.exists()
        assert stored.path.stat().st_nlink == 1

    def test_a_concurrent_start_that_removes_its_temporary_link_mid_read_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        staged = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        os.link(stored.path, staged)
        real_fstat = os.fstat
        seen: list[int] = []

        def fstat_then_the_other_start_unlinks(fd: int) -> os.stat_result:
            info = real_fstat(fd)
            if not seen and stat.S_ISREG(info.st_mode):
                seen.append(info.st_nlink)
                staged.unlink()
            return info

        monkeypatch.setattr(os, "fstat", fstat_then_the_other_start_unlinks)

        assert read_stored_token(tmp_path) == stored.token
        assert seen == [2]

    @pytest.mark.parametrize("foreign", [False, True], ids=["temporary-only", "with-foreign-link"])
    def test_a_temporary_link_removed_after_the_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: bool
    ) -> None:
        stored = load_or_create_token(tmp_path)
        if foreign:
            os.link(stored.path, tmp_path / "copy")
        staged = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        os.link(stored.path, staged)
        scans: list[int] = []

        def scan_then_the_other_start_unlinks(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            yield from owned_files(directory, name)
            scans.append(1)
            staged.unlink(missing_ok=True)

        monkeypatch.setattr(server_auth, "owned_files", scan_then_the_other_start_unlinks)

        if foreign:
            with pytest.raises(TokenFileError, match="hard links"):
                read_stored_token(tmp_path)
            assert (tmp_path / "copy").exists()
        else:
            assert read_stored_token(tmp_path) == stored.token
            assert len(scans) == 1
        assert not staged.exists()

    def test_a_link_count_that_keeps_changing_is_refused_after_a_few_scans(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / ".auth-token.sie-server-0000000000000000")
        scans: list[int] = []

        def scan_then_another_link_appears(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            yield from owned_files(directory, name)
            scans.append(1)
            os.link(stored.path, tmp_path / f".auth-token.sie-server-{len(scans):016x}")

        monkeypatch.setattr(server_auth, "owned_files", scan_then_another_link_appears)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert len(scans) == server_auth._LINK_SCAN_ATTEMPTS

    @pytest.mark.parametrize("foreign", [False, True], ids=["temporary-only", "with-foreign-link"])
    def test_a_temporary_link_listed_twice_by_the_scan_counts_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: bool
    ) -> None:
        stored = load_or_create_token(tmp_path)
        if foreign:
            os.link(stored.path, tmp_path / "copy")
        staged = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        os.link(stored.path, staged)

        def scan_listing_each_entry_twice(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            for found in list(owned_files(directory, name)):
                yield found
                yield found

        monkeypatch.setattr(server_auth, "owned_files", scan_listing_each_entry_twice)

        if foreign:
            with pytest.raises(TokenFileError, match="hard links"):
                read_stored_token(tmp_path)
            assert (tmp_path / "copy").exists()
            assert staged.exists()
        else:
            assert read_stored_token(tmp_path) == stored.token
            assert not staged.exists()

    def test_a_temporary_link_listed_twice_in_a_directory_without_write_permission_counts_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)
        os.link(stored.path, tmp_path / "elsewhere")
        os.link(stored.path, directory / ".auth-token.sie-server-0123456789abcdef")

        def scan_listing_each_entry_twice(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            for found in list(owned_files(directory, name)):
                yield found
                yield found

        monkeypatch.setattr(server_auth, "owned_files", scan_listing_each_entry_twice)
        directory.chmod(0o500)
        try:
            with pytest.raises(TokenFileError, match="hard links"):
                read_stored_token(directory)
        finally:
            directory.chmod(0o700)

    def test_a_temporary_link_renamed_during_the_scan_does_not_hide_a_foreign_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / "copy")
        staged = tmp_path / ".auth-token.sie-server-aaaaaaaaaaaaaaaa"
        renamed = tmp_path / ".auth-token.sie-server-bbbbbbbbbbbbbbbb"
        os.link(stored.path, staged)

        def scan_while_the_link_is_renamed(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            for entry, info in list(owned_files(directory, name)):
                yield entry, info
                if entry == staged:
                    staged.rename(renamed)
                    yield renamed, renamed.lstat()

        monkeypatch.setattr(server_auth, "owned_files", scan_while_the_link_is_renamed)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert (tmp_path / "copy").exists()

    def test_a_temporary_link_added_before_and_removed_after_the_listing_does_not_hide_a_foreign_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / "copy")
        staged = tmp_path / ".auth-token.sie-server-cccccccccccccccc"

        def scan_around_a_passing_link(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            os.link(stored.path, staged)
            found = list(owned_files(directory, name))
            staged.unlink()
            yield from found

        monkeypatch.setattr(server_auth, "owned_files", scan_around_a_passing_link)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert (tmp_path / "copy").exists()

    def test_a_temporary_name_that_no_longer_holds_the_token_is_not_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        staged = tmp_path / ".auth-token.sie-server-eeeeeeeeeeeeeeee"
        os.link(stored.path, staged)

        def scan_then_the_name_is_replaced(directory: Path, name: re.Pattern[str]) -> Iterator[Any]:
            found = list(owned_files(directory, name))
            staged.unlink()
            staged.write_text("unrelated\n")
            os.link(stored.path, tmp_path / ".auth-token.sie-server-ffffffffffffffff")
            yield from found

        monkeypatch.setattr(server_auth, "owned_files", scan_then_the_name_is_replaced)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert staged.read_text() == "unrelated\n"

    def test_a_symbolic_link_named_like_a_temporary_file_does_not_hide_a_foreign_link(self, tmp_path: Path) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / "copy")
        decoy = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        decoy.symlink_to(stored.path)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert decoy.is_symlink()
        assert (tmp_path / "copy").exists()

    def test_a_temporary_link_in_a_directory_without_write_permission_is_tolerated(self, tmp_path: Path) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)
        os.link(stored.path, directory / ".auth-token.sie-server-0123456789abcdef")
        directory.chmod(0o500)
        try:
            assert read_stored_token(directory) == stored.token
        finally:
            directory.chmod(0o700)

    def test_a_temporary_link_that_a_writable_directory_cannot_remove_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        staged = tmp_path / ".auth-token.sie-server-0123456789abcdef"
        os.link(stored.path, staged)
        _refuse_unlink(staged, monkeypatch, errno.EPERM)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert staged.exists()

    def test_a_read_only_directory_does_not_excuse_a_link_kept_by_another_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)
        immutable = directory / ".auth-token.sie-server-0000000000000001"
        os.link(stored.path, immutable)
        os.link(stored.path, directory / ".auth-token.sie-server-0000000000000002")
        _refuse_unlink(immutable, monkeypatch, errno.EPERM)
        directory.chmod(0o500)
        try:
            with pytest.raises(TokenFileError, match="hard links"):
                read_stored_token(directory)
        finally:
            directory.chmod(0o700)

    def test_a_link_added_elsewhere_while_a_read_only_directory_is_cleaned_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)
        os.link(stored.path, directory / ".auth-token.sie-server-0123456789abcdef")
        unlink_token_link = server_auth._unlink_token_link

        def unlink_then_a_link_appears_elsewhere(entry: Path, token: os.stat_result) -> int | None:
            code = unlink_token_link(entry, token)
            os.link(stored.path, tmp_path / "elsewhere")
            return code

        monkeypatch.setattr(server_auth, "_unlink_token_link", unlink_then_a_link_appears_elsewhere)
        directory.chmod(0o500)
        try:
            with pytest.raises(TokenFileError, match="hard links"):
                read_stored_token(directory)
        finally:
            directory.chmod(0o700)

    @pytest.mark.parametrize("name", [".auth-token.bak", ".auth-token.0123456789abcdef"])
    def test_a_users_link_named_like_a_temporary_file_is_kept_and_refused(self, tmp_path: Path, name: str) -> None:
        stored = load_or_create_token(tmp_path)
        backup = tmp_path / name
        os.link(stored.path, backup)

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)
        assert backup.exists()

    def test_only_old_temporary_files_of_the_server_are_cleaned_up(self, tmp_path: Path) -> None:
        old = time.time() - 600
        orphan = tmp_path / ".auth-token.sie-server-00000000000000aa"
        in_flight = tmp_path / ".auth-token.sie-server-00000000000000bb"
        users = tmp_path / ".auth-token.backup"
        for path in (orphan, in_flight, users):
            path.write_text("unused\n")
            path.chmod(0o600)
        for path in (orphan, users):
            os.utime(path, (old, old))

        load_or_create_token(tmp_path)

        assert not orphan.exists()
        assert in_flight.exists()
        assert users.exists()

    def test_an_existing_token_in_a_directory_without_write_permission_is_read(self, tmp_path: Path) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)
        directory.chmod(0o500)
        try:
            again = load_or_create_token(directory)
        finally:
            directory.chmod(0o700)

        assert not again.created
        assert again.token == stored.token

    def test_an_existing_token_is_read_without_creating_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = load_or_create_token(tmp_path)
        creates = _record_creates_in(tmp_path, monkeypatch)

        assert load_or_create_token(tmp_path).token == stored.token
        assert creates == []

    @pytest.mark.parametrize("code", [errno.EACCES, errno.EROFS])
    def test_a_token_that_appears_while_the_directory_refuses_writes_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        def another_start_finished_first() -> None:
            token_path(tmp_path).write_text(f"{TOKEN}\n")
            token_path(tmp_path).chmod(0o600)

        _refuse_creates_in(tmp_path, monkeypatch, code, before=another_start_finished_first)

        stored = load_or_create_token(tmp_path)

        assert not stored.created
        assert stored.token == TOKEN

    def test_a_read_only_volume_without_a_token_names_the_temporary_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _refuse_creates_in(tmp_path, monkeypatch, errno.EROFS)

        with pytest.raises(TokenFileError, match=r"Cannot create .*/\.auth-token\.sie-server-[0-9a-f]{16}: "):
            load_or_create_token(tmp_path)

    def test_a_hard_linked_token_is_refused(self, tmp_path: Path) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / "copy")

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)

    @pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
    def test_a_directory_others_can_write_is_refused(self, tmp_path: Path, mode: int) -> None:
        directory = tmp_path / "shared dir"
        directory.mkdir()
        directory.chmod(mode)

        with pytest.raises(TokenFileError, match=re.escape(f"(chmod 700 '{directory}')")):
            load_or_create_token(directory)
        assert not token_path(directory).exists()

    @_MACOS_ACL
    def test_a_directory_with_an_access_control_list_entry_that_allows_access_is_refused(
        self, tmp_path: Path, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        existing = tmp_path / "Application Support"
        load_or_create_token(existing)
        empty = tmp_path / "empty"
        empty.mkdir(mode=0o700)
        for directory in (existing, empty):
            add_acl_entry(directory, "everyone deny delete")
            add_acl_entry(directory, "everyone allow add_file,add_subdirectory,delete_child")

        with pytest.raises(TokenFileError) as refused:
            read_stored_token(existing)
        quoted = f"'{existing}'"
        message = str(refused.value)
        assert f"List the entries with ls -led {quoted} and remove" in message
        assert f'remove each allow entry with chmod -a "<entry>" {quoted}, giving the entry' in message
        with pytest.raises(TokenFileError, match="entry that allows access"):
            load_or_create_token(empty)
        assert not token_path(empty).exists()

    @_MACOS_ACL
    def test_removing_each_allow_entry_by_its_text_keeps_the_deny_entries(
        self, tmp_path: Path, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        add_acl_entry(tmp_path, "everyone deny delete,directory_inherit")
        add_acl_entry(tmp_path, "everyone allow list,directory_inherit")
        directory = tmp_path / "sie"
        directory.mkdir(mode=0o700)
        add_acl_entry(directory, "user:nobody allow list")
        add_acl_entry(directory, "user:daemon allow search")
        with pytest.raises(TokenFileError, match="without its number or the word inherited"):
            load_or_create_token(directory)

        for entry in _acl_entries(directory):
            if " allow " in entry:
                subprocess.run(  # noqa: S603
                    ["/bin/chmod", "-a", entry.replace(" inherited ", " "), str(directory)], check=True
                )

        assert _acl_entries(directory) == ["group:everyone inherited deny delete,directory_inherit"]
        assert load_or_create_token(directory).created

    @_MACOS_ACL
    def test_deny_entries_like_the_home_folder_default_are_accepted(
        self, tmp_path: Path, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        directory = tmp_path / "home"
        directory.mkdir(mode=0o700)
        add_acl_entry(directory, "everyone deny delete")

        stored = load_or_create_token(directory)
        add_acl_entry(stored.path, "everyone deny delete")

        assert stored.created
        assert read_stored_token(directory) == stored.token

    @_MACOS_ACL
    def test_a_token_file_with_an_access_control_list_entry_that_allows_access_is_refused(
        self, tmp_path: Path, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        directory = tmp_path / "Application Support"
        stored = load_or_create_token(directory)
        add_acl_entry(stored.path, "everyone allow read")

        with pytest.raises(TokenFileError) as refused:
            read_stored_token(directory)
        message = str(refused.value)
        assert f"so others may have read the token. Delete the file (rm '{stored.path}')" in message
        assert f"Removing the entries with chmod -N '{stored.path}' instead keeps the current token." in message

    @_MACOS_ACL
    def test_a_token_file_that_inherited_an_entry_is_refused_after_the_directory_is_cleared(
        self, tmp_path: Path, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        directory = tmp_path / "sie"
        directory.mkdir(mode=0o700)
        add_acl_entry(directory, "everyone allow read,file_inherit")
        token_path(directory).write_text(f"{TOKEN}\n")
        token_path(directory).chmod(0o600)
        subprocess.run(["/bin/chmod", "-N", str(directory)], check=True)  # noqa: S603

        with pytest.raises(TokenFileError, match=re.escape(f"chmod -N {token_path(directory)}")):
            read_stored_token(directory)

    @_MACOS_ACL
    def test_an_access_control_list_that_cannot_be_read_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_read(fd: int, kind: int) -> None:
            ctypes.set_errno(errno.EIO)

        monkeypatch.setattr(server_auth._libsystem(), "acl_get_fd_np", failing_read)

        with pytest.raises(TokenFileError, match="Cannot read the access control list"):
            load_or_create_token(tmp_path)
        assert not token_path(tmp_path).exists()

    @_MACOS_ACL
    def test_an_access_control_list_entry_of_unknown_kind_counts_as_allowing_access(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, add_acl_entry: Callable[[Path, str], None]
    ) -> None:
        add_acl_entry(tmp_path, "everyone deny delete")
        monkeypatch.setattr(server_auth._libsystem(), "acl_get_tag_type", lambda entry, tag: -1)

        with pytest.raises(TokenFileError, match="entry that allows access"):
            load_or_create_token(tmp_path)

    def test_a_directory_of_another_user_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        load_or_create_token(tmp_path)
        monkeypatch.setattr(os, "getuid", lambda: tmp_path.stat().st_uid + 1)

        with pytest.raises(TokenFileError, match="must be owned by the user"):
            read_stored_token(tmp_path)

    def test_the_worker_id_is_never_written_through_a_symbolic_link(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        (tmp_path / "worker-id").symlink_to(target)

        _get_or_create_worker_id(str(tmp_path))

        assert not target.exists()

    def test_a_symbolic_link_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        target.write_text(f"{TOKEN}\n")
        target.chmod(0o600)
        token_path(tmp_path).symlink_to(target)

        with pytest.raises(TokenFileError, match="symbolic link"):
            load_or_create_token(tmp_path)

    @pytest.mark.parametrize("content", ["", "\n", "two tokens\n", "a,b\n", "tokén\n"])
    def test_malformed_content_is_refused(self, tmp_path: Path, content: str) -> None:
        path = token_path(tmp_path)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

        with pytest.raises(TokenFileError, match="must hold one token"):
            read_stored_token(tmp_path)

    def test_server_tokens_prefer_the_configured_ones(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))
        load_or_create_token(tmp_path)

        assert server_tokens({"SIE_AUTH_TOKEN": TOKEN}) == (TOKEN,)

    def test_server_tokens_fall_back_to_the_persisted_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))
        stored = load_or_create_token(tmp_path)

        assert server_tokens(os.environ) == (stored.token,)

    def test_server_tokens_never_generate_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        with pytest.raises(TokenFileError):
            server_tokens(os.environ)
        assert not token_path(tmp_path).exists()


def _plain_app() -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> str:
        return "ok"

    @app.get("/v1/thing")
    def thing() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws/status")
    async def status(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"ready": True})
        await websocket.close()

    return app


def _protected_app(tokens: tuple[str, ...] = (TOKEN, OTHER)) -> FastAPI:
    app = _plain_app()
    app.add_middleware(ServerAuthMiddleware, tokens=tokens)
    return app


class TestMiddleware:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_protected_app())

    def test_probes_need_no_token(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200

    def test_a_trailing_slash_is_not_a_probe(self, client: TestClient) -> None:
        assert client.get("/healthz/").status_code == 401

    def test_missing_credentials(self, client: TestClient) -> None:
        response = client.get("/v1/thing")

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["detail"]["code"] == "UNAUTHORIZED"

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": f"Bearer {TOKEN}"},
            {"Authorization": f"bearer {TOKEN}"},
            {"Authorization": f"  Bearer   {TOKEN}  "},
            {"Authorization": TOKEN},
            {"x-api-key": TOKEN},
            {"X-API-Key": OTHER},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": TOKEN},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": OTHER},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": "wrong"},
            {"Authorization": "", "x-api-key": TOKEN},
        ],
    )
    def test_accepted_credentials(self, client: TestClient, headers: dict[str, str]) -> None:
        assert client.get("/v1/thing", headers=headers).status_code == 200

    @pytest.mark.parametrize(
        ("headers", "message"),
        [
            ({"Authorization": "Bearer wrong"}, "Invalid token"),
            ({"x-api-key": "wrong"}, "Invalid token"),
            ({"Authorization": f"Bearer {TOKEN[:-1]}"}, "Invalid token"),
            ({"Authorization": f"Basic {TOKEN}"}, "Invalid token"),
            ({"Authorization": "Bearer"}, "Invalid token"),
            ({"Authorization": "", "x-api-key": ""}, "Missing credentials"),
            ({"Authorization": "Bearer wrong", "x-api-key": TOKEN}, "Invalid token"),
        ],
    )
    def test_refused_credentials(self, client: TestClient, headers: dict[str, str], message: str) -> None:
        response = client.get("/v1/thing", headers=headers)

        assert response.status_code == 401
        assert message in response.json()["detail"]["message"]

    @pytest.mark.parametrize(
        "headers",
        [
            [("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {OTHER}")],
            [("Authorization", f"Bearer {TOKEN}"), ("Authorization", "Bearer wrong"), ("x-api-key", TOKEN)],
            [("x-api-key", TOKEN), ("x-api-key", OTHER)],
        ],
    )
    def test_a_repeated_header_that_disagrees_is_refused(
        self, client: TestClient, headers: list[tuple[str, str]]
    ) -> None:
        # One header repeated with two tokens is not the two-header case, so it stays a refusal.
        response = client.get("/v1/thing", headers=headers)

        assert response.status_code == 401
        assert "Repeated credential headers carry different tokens" in response.json()["detail"]["message"]

    def test_websocket_without_a_token_is_denied(self, client: TestClient) -> None:
        with pytest.raises(WebSocketDenialResponse) as denied, client.websocket_connect("/ws/status"):
            pass
        assert denied.value.status_code == 401

    def test_websocket_with_a_token_connects(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/status", headers={"Authorization": f"Bearer {TOKEN}"}) as websocket:
            assert websocket.receive_json() == {"ready": True}

    def test_a_middleware_without_tokens_cannot_be_built(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            TestClient(_protected_app(tokens=())).get("/healthz")


class TestRequestCredentials:
    def test_authorization_decides_when_both_headers_are_sent(self) -> None:
        headers = [(b"authorization", b"Bearer first"), (b"x-api-key", b"second")]
        assert request_credentials(headers) == {b"first"}

    def test_x_api_key_alone_decides(self) -> None:
        assert request_credentials([(b"x-api-key", b" second ")]) == {b"second"}

    def test_an_empty_authorization_leaves_x_api_key_to_decide(self) -> None:
        assert request_credentials([(b"authorization", b""), (b"x-api-key", b"second")]) == {b"second"}


class TestHostAllowlist:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("localhost", "localhost"),
            ("LOCALHOST:8080", "localhost"),
            ("127.0.0.1:8080", "127.0.0.1"),
            ("[::1]:8080", "[::1]"),
            ("[::1]", "[::1]"),
            ("evil.example", "evil.example"),
            ("[::1", None),
            ("localhost:80x", None),
        ],
    )
    def test_request_host(self, host: str, expected: str | None) -> None:
        assert request_host([(b"host", host.encode())]) == expected

    def test_a_missing_host_matches_nothing(self) -> None:
        assert request_host([(b"accept", b"*/*")]) is None

    @pytest.mark.parametrize(
        ("bind", "extra"), [("127.0.0.1", None), ("localhost", None), ("::1", None), ("127.0.0.2", "127.0.0.2")]
    )
    def test_allowlist_holds_the_loopback_names_and_the_bind_address(self, bind: str, extra: str | None) -> None:
        allowed = loopback_host_allowlist(bind)
        assert {"localhost", "127.0.0.1", "[::1]"} <= allowed
        if extra:
            assert extra in allowed


def _loopback_client(bind_host: str, base_url: str = "http://localhost:8080", **config: object) -> TestClient:
    return TestClient(AppFactory.create_app(AppStateConfig(bind_host=bind_host, **config)), base_url=base_url)


class TestLoopbackServerApp:
    @pytest.mark.parametrize("base_url", ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost"])
    def test_loopback_clients_are_answered_without_a_warning(
        self, base_url: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _loopback_client("127.0.0.1", base_url)

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            assert client.get("/openapi.json").status_code == 200
            assert client.get("/healthz").status_code == 200
        assert _host_warnings(caplog) == []

    def test_an_ipv6_loopback_host_is_answered(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _loopback_client("::1")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            assert client.get("/openapi.json", headers={"Host": "[::1]:8080"}).status_code == 200
        assert _host_warnings(caplog) == []

    @pytest.mark.parametrize(
        "host", ["evil.example", "localhost.evil.example", "evil.example:8080", "0.0.0.0:8080", "host.docker.internal"]
    )
    def test_another_host_is_answered_with_one_warning(self, host: str, caplog: pytest.LogCaptureFixture) -> None:
        client = _loopback_client("127.0.0.1")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            assert client.get("/openapi.json", headers={"Host": host}).status_code == 200
            assert client.get("/healthz", headers={"Host": host}).status_code == 200

        (warning,) = _host_warnings(caplog)
        assert "0.13.0" in warning
        assert repr(host.partition(":")[0]) in warning

    def test_another_host_on_a_websocket_connects_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        app = _plain_app()
        app.add_middleware(LoopbackHostMiddleware, allowed_hosts=loopback_host_allowlist("127.0.0.1"))
        client = TestClient(app)

        with (
            caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER),
            client.websocket_connect("/ws/status", headers={"Host": "evil.example"}) as websocket,
        ):
            assert websocket.receive_json() == {"ready": True}
        (warning,) = _host_warnings(caplog)
        assert "0.13.0" in warning

    def test_a_long_host_is_remembered_and_logged_by_its_first_characters(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _loopback_client("127.0.0.1")
        prefix = "a" * 100

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            for tail in ("b" * 5000, "c" * 5000):
                assert client.get("/healthz", headers={"Host": prefix + tail}).status_code == 200

        (warning,) = _host_warnings(caplog)
        assert repr(prefix) in warning
        assert "a" * 101 not in warning
        assert "bbbb" not in warning

    def test_each_other_host_is_warned_about_once(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _loopback_client("127.0.0.1")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            for host in ("a.example", "b.example", "a.example", "b.example"):
                assert client.get("/healthz", headers={"Host": host}).status_code == 200

        assert len(_host_warnings(caplog)) == 2

    def test_warnings_stop_after_many_hosts(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _loopback_client("127.0.0.1")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            for index in range(80):
                client.get("/healthz", headers={"Host": f"host-{index}.example"})

        assert len(_host_warnings(caplog)) == 64

    def test_the_bind_address_is_answered(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _loopback_client("127.0.0.2", "http://127.0.0.2:8080")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            assert client.get("/openapi.json").status_code == 200
        assert _host_warnings(caplog) == []

    @pytest.mark.parametrize("bind_host", ["0.0.0.0", "192.168.1.20", None])  # noqa: S104
    def test_an_exposed_or_unknown_bind_is_not_host_checked(
        self, bind_host: str | None, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = TestClient(AppFactory.create_app(AppStateConfig(bind_host=bind_host)))

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            assert client.get("/openapi.json", headers={"Host": "evil.example"}).status_code == 200
        assert _host_warnings(caplog) == []

    def test_token_auth_replaces_the_host_check(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        client = _loopback_client("127.0.0.1", auth_mode="token")

        with caplog.at_level(logging.WARNING, logger=_AUTH_LOGGER):
            response = client.get("/openapi.json", headers={"Host": "evil.example", "x-api-key": TOKEN})
        assert response.status_code == 200
        assert _host_warnings(caplog) == []


class TestLoopbackHostEnforcement:
    @pytest.fixture
    def client(self) -> TestClient:
        app = _plain_app()
        app.add_middleware(LoopbackHostMiddleware, allowed_hosts=loopback_host_allowlist("127.0.0.2"), enforce=True)
        return TestClient(app, base_url="http://127.0.0.2:8080")

    def test_the_allowed_hosts_are_answered(self, client: TestClient) -> None:
        for host in ("localhost", "127.0.0.1:8080", "[::1]", "127.0.0.2:8080"):
            assert client.get("/v1/thing", headers={"Host": host}).status_code == 200

    def test_another_host_is_refused_with_the_allowed_names(self, client: TestClient) -> None:
        response = client.get("/v1/thing", headers={"Host": "evil.example"})

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "INVALID_INPUT",
            "message": "This server listens on loopback and answers only requests addressed to "
            "localhost, 127.0.0.1, [::1] or 127.0.0.2.",
        }

    def test_another_host_on_a_websocket_is_refused(self, client: TestClient) -> None:
        with (
            pytest.raises(WebSocketDenialResponse) as denied,
            client.websocket_connect("/ws/status", headers={"Host": "evil.example"}),
        ):
            pass
        assert denied.value.status_code == 400


class TestWebSocketCloseFallback:
    """Without the ``websocket.http.response`` extension a refusal can only close the socket."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda inner: ServerAuthMiddleware(inner, tokens=(TOKEN,)),
            lambda inner: LoopbackHostMiddleware(inner, loopback_host_allowlist("127.0.0.1"), enforce=True),
        ],
        ids=["token", "host"],
    )
    @pytest.mark.parametrize("extensions", [None, {}], ids=["absent", "empty"])
    def test_a_refused_websocket_is_closed_with_policy_violation(
        self, build: Callable[[Any], Any], extensions: dict[str, Any] | None
    ) -> None:
        inner_calls: list[dict[str, Any]] = []

        async def inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
            inner_calls.append(scope)

        scope: dict[str, Any] = {"type": "websocket", "path": "/ws/status", "headers": [(b"host", b"evil.example")]}
        if extensions is not None:
            scope["extensions"] = extensions
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "websocket.connect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        asyncio.run(build(inner)(scope, receive, send))

        assert inner_calls == []
        assert sent == [{"type": "websocket.close", "code": 1008}]


class TestServerApp:
    def test_token_mode_protects_the_real_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        client = TestClient(AppFactory.create_app(AppStateConfig(auth_mode="token")))

        assert client.get("/healthz").status_code == 200
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/openapi.json", headers={"x-api-key": TOKEN}).status_code == 200
        with pytest.raises(WebSocketDenialResponse), client.websocket_connect("/ws/status"):
            pass

    def test_default_mode_installs_no_auth(self) -> None:
        client = TestClient(AppFactory.create_app(AppStateConfig()))

        assert client.get("/openapi.json").status_code == 200

    def test_token_mode_without_any_token_refuses_to_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        with pytest.raises(TokenFileError):
            AppFactory.create_app(AppStateConfig(auth_mode="token"))


class TestAppStateConfigRoundTrip:
    def test_the_resolved_mode_survives_the_factory_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        AppStateConfig(auth_mode="token").save_to_env_vars()

        assert os.environ["SIE_AUTH_MODE"] == "token"
        assert AppStateConfig.from_env_vars().auth_mode == "token"

    def test_an_unresolved_auto_requires_a_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_MODE", "auto")

        assert AppStateConfig.from_env_vars().auth_mode == "token"

    def test_the_bind_address_survives_the_factory_boundary(self) -> None:
        AppStateConfig(bind_host="127.0.0.1").save_to_env_vars()

        assert AppStateConfig.from_env_vars().bind_host == "127.0.0.1"

    def test_the_factory_refuses_a_blank_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", " , ")

        with pytest.raises(AuthConfigError, match="holds no token"):
            AppStateConfig.from_env_vars()


_FIRST_START_SCRIPT = (
    "import sys; from pathlib import Path; "
    "from sie_server.core.server_auth import load_or_create_token; "
    "print(load_or_create_token(Path(sys.argv[1])).token)"
)


_AUTH_LOGGER = "sie_server.core.server_auth"


def _host_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _AUTH_LOGGER and record.levelno == logging.WARNING and "Host" in record.getMessage()
    ]


def _creates_in(directory: Path, path: Any, flags: int) -> bool:
    return bool(flags & os.O_CREAT) and Path(os.fsdecode(path)).parent == directory


def _record_creates_in(directory: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    creates: list[str] = []
    real_open = os.open

    def recording_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if _creates_in(directory, path, flags):
            creates.append(os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    return creates


def _refuse_creates_in(
    directory: Path, monkeypatch: pytest.MonkeyPatch, code: int, before: Callable[[], None] | None = None
) -> None:
    """Make creating a file in ``directory`` fail as a read-only directory or volume does."""
    real_open = os.open

    def refusing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if _creates_in(directory, path, flags):
            if before is not None:
                before()
            raise OSError(code, os.strerror(code), os.fsdecode(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refusing_open)


def _refuse_unlink(target: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    """Make unlinking ``target`` fail with ``code``, as an immutable file makes it fail with EPERM."""
    real_unlink = Path.unlink

    def refusing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == target:
            raise OSError(code, os.strerror(code), str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refusing_unlink)


def _acl_entries(path: Path) -> list[str]:
    listing = subprocess.run(  # noqa: S603
        ["/bin/ls", "-led", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return [line.split(": ", 1)[1] for line in listing.splitlines()[1:]]


def _capture(call: Callable[[], object], outcome: list[BaseException]) -> None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001
        outcome.append(exc)
