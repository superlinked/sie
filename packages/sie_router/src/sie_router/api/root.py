from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# Load static HTML file at module load time
_index_html = files("sie_router.static").joinpath("index.html").read_text()


@router.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Basic HTML status page for the root endpoint."""
    return HTMLResponse(content=_index_html)
