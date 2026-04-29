from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version


def _get_config_version() -> str:
    try:
        return pkg_version("sie-config")
    except PackageNotFoundError:
        return "unknown"


CONFIG_VERSION = _get_config_version()
