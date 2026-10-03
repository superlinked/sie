"""FastMCP server exposing document tools.

The cluster client is created once per process in the lifespan and reused across
requests.
"""

import base64
import binascii
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from sie_sdk import SIEAsyncClient

from sie_mcp import describe, jobs, offload, qa, structured
from sie_mcp.config import MCPConfig

logger = logging.getLogger(__name__)


def _decode_base64_text(content_base64: str, max_bytes: int) -> str:
    try:
        raw = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 must be valid base64") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"decoded content exceeds {max_bytes} bytes")
    return raw.decode("utf-8", errors="replace")


def _validate_plain_text_size(content: str, max_bytes: int) -> str:
    if len(content.encode("utf-8")) > max_bytes:
        raise ValueError(f"content exceeds {max_bytes} bytes")
    return content


async def _resolve_text_input(
    client: Any,
    config: MCPConfig,
    *,
    content: str | None,
    content_base64: str | None,
    document_base64: str | None,
    filename: str | None,
    engine: jobs.EngineName,
) -> tuple[str, dict[str, Any]]:
    provided = [content is not None, content_base64 is not None, document_base64 is not None]
    if sum(provided) != 1:
        raise ValueError("provide exactly one of content, content_base64, or document_base64")
    if content is not None:
        return _validate_plain_text_size(content, config.max_document_bytes), {"input_type": "content"}
    if content_base64 is not None:
        return _decode_base64_text(content_base64, config.max_document_bytes), {"input_type": "content_base64"}

    assert document_base64 is not None
    result = await jobs.docs_to_markdown(
        client,
        document_base64=document_base64,
        filename=filename,
        ocr=False,
        engine=engine,
        model=config.docling_model,
        vlocr_model=config.vlocr_model,
        gpu=config.docs_gpu,
        max_document_bytes=config.max_document_bytes,
    )
    return result["markdown"], {"input_type": "document_base64", "document_metadata": result["metadata"]}


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    config = MCPConfig.from_env()
    client = SIEAsyncClient(
        config.sie_base_url,
        api_key=config.sie_api_key,
        timeout_s=config.timeout_s,
    )
    logger.info("sie-mcp connected to cluster at %s", config.sie_base_url)
    try:
        yield {"client": client, "config": config}
    finally:
        await client.close()


def _transport_security(config: MCPConfig) -> TransportSecuritySettings:
    """Build FastMCP's DNS-rebinding (Host/Origin) protection settings.

    FastMCP auto-enables a *localhost-only* allow-list whenever its host is the
    default ``127.0.0.1``, which would reject the documented remote ``https://<host>/mcp``
    endpoint with ``421 Invalid Host``. This edge is a remote service fronted by TLS
    ingress and guarded by its own connector-secret auth, so:

    - With ``SIE_MCP_ALLOWED_HOSTS`` set, keep protection on, scoped to those hosts.
    - Without it, disable Host validation here (auth is the access control).
    """
    if config.allowed_hosts:
        origins = [origin for host in config.allowed_hosts for origin in (f"https://{host}", f"http://{host}")]
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=config.allowed_hosts,
            allowed_origins=origins,
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_server(config: MCPConfig) -> FastMCP:
    server = FastMCP(
        "sie-mcp",
        stateless_http=True,
        lifespan=_lifespan,
        transport_security=_transport_security(config),
    )

    @server.tool()
    async def docs_to_markdown(
        ctx: Context,
        document_base64: str,
        filename: str | None = None,
        ocr: bool = False,
        engine: jobs.EngineName = "auto",
    ) -> dict[str, Any]:
        """Convert a document (PDF/DOCX/PPTX/XLSX/HTML/scan) to clean markdown on the SIE cluster.

        Send the document's raw bytes base64-encoded; you get markdown back. Read the
        source as bytes — do NOT view or attach the document in the conversation — then
        operate on the returned markdown so the page-image tokens are never billed.

        Args:
            document_base64: The document's raw bytes, base64-encoded.
            filename: Optional original filename; its extension hints the format.
            ocr: Turn on Docling's built-in OCR. Only applies under engine="docling";
                under "auto" the VL-OCR fallback handles scans, so this is ignored
                (Docling's built-in OCR is the baseline VL-OCR supersedes).
            engine: Conversion engine. "auto" (default) runs Docling and falls back to
                VL-OCR for scanned/image-only pages; "docling" forces the Docling front
                door; "vl-ocr" rasterizes the document and runs the VL-OCR model per page
                (best for scanned or complex-layout PDFs).
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        result = await jobs.docs_to_markdown(
            client,
            document_base64=document_base64,
            filename=filename,
            ocr=ocr,
            engine=engine,
            model=config.docling_model,
            vlocr_model=config.vlocr_model,
            gpu=config.docs_gpu,
            max_document_bytes=config.max_document_bytes,
        )
        return {"markdown": result["markdown"], "metadata": result["metadata"]}

    @server.tool()
    async def answer_questions(
        ctx: Context,
        documents: list[str],
        questions: list[str],
    ) -> dict[str, Any]:
        """Answer questions grounded in a document set, on the SIE cluster.

        For a document set too large to drop whole into the conversation: pass the
        documents' text and your questions, and get back an answer per question
        plus the passages it was grounded in. Read large sources as text (e.g. the
        markdown from ``docs_to_markdown``) and answer through this tool rather
        than pasting the whole corpus — only the retrieved passages are billed.

        Retrieval is transient: the documents are chunked, encoded, reranked, and
        answered over per call. Nothing is persisted between calls and no standing
        index is built, so pass the relevant documents in each request.

        Args:
            documents: The source documents' text (plain text or markdown).
            questions: The questions to answer over the documents.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        result = await qa.answer_questions(
            client,
            documents=documents,
            questions=questions,
            encode_model=config.encode_model,
            rerank_model=config.rerank_model,
            generate_model=config.generate_model,
            top_k=config.qa_top_k,
            rerank_candidates=config.qa_rerank_candidates,
            chunk_chars=config.qa_chunk_chars,
            chunk_overlap_chars=config.qa_chunk_overlap_chars,
            max_tokens=config.qa_max_tokens,
            max_document_chars=config.qa_max_document_chars,
            max_questions=config.qa_max_questions,
            max_chunks=config.qa_max_chunks,
            gpu=config.qa_gpu,
        )
        return {"answers": result["answers"]}

    @server.tool()
    async def summarize_document(
        ctx: Context,
        content: str | None = None,
        content_base64: str | None = None,
        document_base64: str | None = None,
        filename: str | None = None,
        engine: jobs.EngineName = "auto",
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Summarize a document/text input on the SIE cluster.

        Provide exactly one of ``content``, ``content_base64``, or ``document_base64``.
        For source documents, pass raw file bytes as ``document_base64``; the edge first
        converts them through ``docs_to_markdown`` and then summarizes the markdown.

        Args:
            content: Plain text/markdown content. Prefer ``content_base64`` for large files
                so the calling model does not read the content directly.
            content_base64: UTF-8 text/markdown bytes, base64-encoded.
            document_base64: Source document bytes, base64-encoded.
            filename: Optional source filename; used when ``document_base64`` is set.
            engine: Document conversion engine for ``document_base64`` inputs.
            model: Optional generation model override.
            max_output_tokens: Optional summary token ceiling.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        text, input_metadata = await _resolve_text_input(
            client,
            config,
            content=content,
            content_base64=content_base64,
            document_base64=document_base64,
            filename=filename,
            engine=engine,
        )
        result = await offload.summarize_document(
            client,
            content=text,
            model=model or config.generate_model,
            gpu=config.generate_gpu,
            max_output_tokens=max_output_tokens or offload.SUMMARY_REDUCE_MAX_TOKENS,
        )
        result["metadata"] = {**input_metadata, **result["metadata"]}
        return result

    @server.tool()
    async def extract_entities(
        ctx: Context,
        labels: list[str],
        content: str | None = None,
        content_base64: str | None = None,
        document_base64: str | None = None,
        filename: str | None = None,
        engine: jobs.EngineName = "auto",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Extract zero-shot entities from a document/text input on the SIE cluster.

        Provide labels such as ``["person", "organization", "date", "amount"]`` and
        exactly one of ``content``, ``content_base64``, or ``document_base64``. Document
        inputs are converted to markdown first.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        text, input_metadata = await _resolve_text_input(
            client,
            config,
            content=content,
            content_base64=content_base64,
            document_base64=document_base64,
            filename=filename,
            engine=engine,
        )
        result = await offload.extract_entities(
            client,
            content=text,
            labels=labels,
            model=model or config.extract_model,
            gpu=config.extract_gpu,
        )
        result["metadata"] = {**input_metadata, **result["metadata"]}
        return result

    @server.tool()
    async def redact_pii(
        ctx: Context,
        content: str | None = None,
        content_base64: str | None = None,
        document_base64: str | None = None,
        filename: str | None = None,
        engine: jobs.EngineName = "auto",
        labels: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Redact PII from a document/text input on the SIE cluster.

        The tool returns redacted text and counts, but intentionally does not return the
        placeholder-to-original map. That preserves the MCP privacy contract: original PII
        is not handed back to the calling model.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        text, input_metadata = await _resolve_text_input(
            client,
            config,
            content=content,
            content_base64=content_base64,
            document_base64=document_base64,
            filename=filename,
            engine=engine,
        )
        result = await offload.redact_pii(
            client,
            content=text,
            labels=labels,
            model=model or config.pii_model,
            gpu=config.extract_gpu,
        )
        result["metadata"] = {**input_metadata, **result["metadata"]}
        return result

    @server.tool()
    async def describe_image(
        ctx: Context,
        image_base64: str,
        labels: list[str] | None = None,
        detailed: bool = False,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Describe an image on the SIE cluster: a caption plus top-k zero-shot tags.

        Send the image's raw bytes base64-encoded; you get back a caption (Florence-2)
        and a list of ``{label, score}`` tags. Read the image as bytes — do NOT view or
        attach it in the conversation — so the image tokens are never billed to the
        calling model. Tagging is zero-shot: the image and the candidate labels are
        embedded (SigLIP/CLIP) and the top-k labels by similarity are returned.

        Args:
            image_base64: The image's raw bytes (JPEG/PNG), base64-encoded.
            labels: Optional candidate labels for zero-shot tagging; defaults to the
                service's configured label set.
            detailed: Use Florence-2 <DETAILED_CAPTION> for a longer caption.
            top_k: How many top tags to return (defaults to the service config).
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        result = await describe.describe_image(
            client,
            image_base64=image_base64,
            labels=labels if labels is not None else config.image_labels,
            caption_model=config.caption_model,
            embed_model=config.embed_model,
            detailed=detailed,
            top_k=top_k if top_k is not None else config.image_top_k,
            gpu=config.image_gpu,
            max_image_bytes=config.max_image_bytes,
        )
        return {"caption": result["caption"], "tags": result["tags"]}

    @server.tool()
    async def extract_structured(
        ctx: Context,
        content: str,
        output_schema: dict[str, Any],
        instruction: str | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Extract schema-valid JSON grounded in document content, on the SIE cluster.

        The model is constrained to ``output_schema`` at decode time and the
        returned ``data`` is validated against the schema before it is handed
        back — a non-conforming response (e.g. if the serving profile bypasses
        grammar enforcement) raises a clear error rather than returning
        "mostly-JSON". Extraction is grounded in ``content``: values are drawn
        from it, not invented. Use this to turn already-extracted text/markdown
        (e.g. the output of ``docs_to_markdown``) into a structured record.

        ``output_schema`` must be in the Outlines-supported subset: no ``$ref``
        (so no recursion), no conditionals (``if``/``then``/``else``), nesting
        depth ≤ 16. Out-of-subset schemas are rejected with a clear error.

        Args:
            content: The source text to extract from (plain text or markdown).
            output_schema: A JSON Schema describing the record to extract.
            instruction: Optional extra guidance on what to pull out.
            model: Optional generation model override (defaults to the cluster's
                configured structured-output model).
            max_output_tokens: Optional output-token ceiling (defaults to the
                edge's ``SIE_MCP_MAX_OUTPUT_TOKENS``); raise it for large records.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        data = await structured.extract_structured(
            client,
            content=content,
            output_schema=output_schema,
            instruction=instruction,
            model=model or config.generate_model,
            gpu=config.generate_gpu,
            max_completion_tokens=max_output_tokens or config.max_output_tokens,
        )
        return {"data": data}

    @server.tool()
    async def generate_structured(
        ctx: Context,
        prompt: str,
        response_format: dict[str, Any],
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate output constrained by a schema / regex / grammar, on the SIE cluster.

        ``response_format`` is the OpenAI-compatible constraint, one of:

        - ``{"type": "json_schema", "json_schema": {"name": ..., "schema": <schema>, "strict": true}}``
        - ``{"type": "json_object"}``
        - ``{"type": "regex", "regex": "<pattern>"}``
        - ``{"type": "grammar", "grammar": "<ebnf>", "syntax": "ebnf"}`` — xgrammar-backed
          models only; rejected up front for the Outlines-backed default model.

        A ``json_schema`` is validated against the Outlines subset before the
        call and the returned JSON is validated against it afterwards. Returns
        the constrained ``content`` string (a JSON string for the json modes).

        Args:
            prompt: The instruction the model should answer under the constraint.
            response_format: The output constraint (see above).
            model: Optional generation model override (defaults to the cluster's
                configured structured-output model).
            max_output_tokens: Optional output-token ceiling (defaults to the
                edge's ``SIE_MCP_MAX_OUTPUT_TOKENS``); raise it for large output.
        """
        lifespan_ctx = ctx.request_context.lifespan_context
        client = lifespan_ctx["client"]
        config: MCPConfig = lifespan_ctx["config"]
        content = await structured.generate_structured(
            client,
            prompt=prompt,
            response_format=response_format,
            model=model or config.generate_model,
            gpu=config.generate_gpu,
            max_completion_tokens=max_output_tokens or config.max_output_tokens,
        )
        return {"content": content}

    return server
