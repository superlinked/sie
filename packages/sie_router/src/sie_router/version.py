from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

SERVER_VERSION_HEADER = "X-SIE-Server-Version"
SDK_VERSION_HEADER = "X-SIE-SDK-Version"


def _get_router_version() -> str:
    try:
        return pkg_version("sie-router")
    except PackageNotFoundError:
        return "unknown"


ROUTER_VERSION = _get_router_version()
