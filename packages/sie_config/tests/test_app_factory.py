from __future__ import annotations

from importlib.metadata import version

import pytest
from sie_config.app_factory import AppFactory


def test_app_reports_the_installed_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sie_config.app_factory.setup_managed_metrics", lambda: None)

    assert AppFactory.create_app().version == version("sie-config")
