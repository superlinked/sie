/**
 * Minimal SSE (Server-Sent Events) parser for the SIE TS SDK.
 *
 * Reads from a `ReadableStreamDefaultReader<Uint8Array>` (a `fetch` response
 * body), decodes the UTF-8 byte stream, and yields each event's `data:`
 * payload as a string. Sentinel `data: [DONE]` payloads close the generator
 * without yielding. The `signal` parameter cancels the read cooperatively
 * and propagates as a `SIEConnectionError`.
 *
 * Scope: this parser supports only the subset of the SSE spec the SIE
 * gateway emits today —
 *
 *   - `data: <single-line>\n\n`
 *   - the literal `data: [DONE]\n\n` terminator
 *   - keep-alive comment lines (`: <text>\n`), which are skipped
 *
 * Other SSE features (`event:`, `id:`, `retry:`, multi-line `data:`
 * continuations) are not produced by `sse.rs` and are not handled here.
 * If the gateway grows new event shapes, extend this parser deliberately
 * rather than adding a generic SSE dependency.
 */

import { SIEConnectionError, SIEStreamError } from "./errors.js";

const SSE_DONE = "[DONE]";

/** Track generation completion separately from a clean HTTP body EOF. */
export class SSECompletion {
  done = false;
  private nativeTerminal = false;
  private readonly chatChoices: number | undefined;
  private readonly finishedChoices = new Set<number>();

  constructor(chatChoices?: number) {
    this.chatChoices =
      chatChoices === undefined
        ? undefined
        : Number.isSafeInteger(chatChoices) && chatChoices > 0
          ? chatChoices
          : 1;
  }

  observe(chunk: unknown): void {
    if (typeof chunk !== "object" || chunk === null) return;
    const record = chunk as Record<string, unknown>;
    if (this.chatChoices === undefined) {
      if (record.done === true) this.nativeTerminal = true;
      return;
    }
    if (!Array.isArray(record.choices)) return;
    for (const choice of record.choices) {
      if (typeof choice !== "object" || choice === null) continue;
      const { index, finish_reason: finish } = choice;
      if (
        typeof index === "number" &&
        Number.isInteger(index) &&
        index >= 0 &&
        index < this.chatChoices &&
        typeof finish === "string" &&
        finish.length > 0
      ) {
        this.finishedChoices.add(index);
      }
    }
  }

  requireComplete(): void {
    if (this.done || this.nativeTerminal) return;
    if (this.chatChoices !== undefined && this.finishedChoices.size === this.chatChoices) return;
    throw new SIEConnectionError(
      "Connection lost during stream: response ended before generation completed",
      "other",
    );
  }
}

/**
 * Hard cap on the in-progress event buffer.
 *
 * The buffer holds bytes received but not yet terminated by an event
 * separator (`\n\n` / `\r\n\r\n`). A well-behaved gateway flushes complete
 * events promptly, so this only grows unbounded if a peer (or a broken
 * intermediary) sends a frame that never terminates. Without a cap that
 * OOMs the client; we instead surface a `SIEStreamError` once the buffer
 * exceeds this size.
 *
 * The bound is measured in UTF-16 code units (the units of
 * `String.prototype.length`), NOT bytes, because the cap is compared
 * against the decoded `buffer.length` below. 8 Mi chars is far above any
 * legitimate single SSE event the SIE gateway emits (chunks are
 * token-sized) while still bounding memory.
 */
const MAX_SSE_BUFFER_CHARS = 8 * 1024 * 1024;

/**
 * Async-iterate over the `data:` payloads of an SSE response body.
 *
 * @param reader     The locked reader returned by `response.body.getReader()`.
 * @param signal     Optional `AbortSignal`; when fired, the generator throws
 *                   `SIEConnectionError` (kind `"other"`) and releases the
 *                   reader so the upstream fetch is cancelled.
 * @param completion Optional generation tracker; records explicit `[DONE]`
 *                   separately from transport EOF for the consuming client.
 * @returns          A generator of `data:` payload strings. The generator
 *                   completes (without throwing) when the server emits
 *                   `[DONE]` or the underlying stream is closed cleanly.
 */
export async function* parseSseStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal?: AbortSignal,
  completion?: SSECompletion,
): AsyncGenerator<string, void, undefined> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  // Tracks whether the loop reached a *clean* terminus — either the
  // `[DONE]` sentinel or a graceful EOF (`result.done`). Any other exit
  // (a caller `break`/early `return` out of the consuming `for await`, a
  // thrown `SIEStreamError`, or a JSON parse `RequestError` raised by the
  // consumer) leaves this `false`, and the `finally` then `cancel()`s the
  // reader so the underlying HTTP body/socket closes promptly. Without
  // that, the gateway never sees the client disconnect and the worker
  // keeps generating to full `max_new_tokens` (wasted / billable tokens).
  let completedCleanly = false;

  // Bridge the AbortSignal onto the reader: cancelling the reader
  // tears down the fetch body promptly. We register the listener
  // once and clean it up in `finally`.
  const onAbort = () => {
    // `cancel()` returns a promise; we don't await it here — the
    // top-level loop sees `signal.aborted` and throws.
    reader.cancel().catch(() => {
      // Cancelling an already-released reader throws; ignore.
    });
  };
  if (signal) {
    if (signal.aborted) {
      throw new SIEConnectionError("Stream aborted before first read", "other");
    }
    signal.addEventListener("abort", onAbort, { once: true });
  }

  try {
    while (true) {
      if (signal?.aborted) {
        throw new SIEConnectionError("Stream aborted by caller", "other");
      }

      // Race `read()` against the abort signal. We can't rely on
      // `reader.cancel()` to settle the pending read promptly across
      // every WHATWG-streams implementation (Node 22's lets an
      // in-flight source `pull()` run to completion), so we drop our
      // own claim on the read result if the caller aborts and surface
      // `SIEConnectionError` immediately. The `reader.cancel()` call
      // in `onAbort` still fires to tear down the upstream fetch.
      let result: Awaited<ReturnType<typeof reader.read>>;
      try {
        if (signal) {
          if (signal.aborted) {
            throw new SIEConnectionError("Stream aborted by caller", "other");
          }
          result = await new Promise<Awaited<ReturnType<typeof reader.read>>>((resolve, reject) => {
            let settled = false;
            const onAbortRace = () => {
              if (settled) return;
              settled = true;
              signal.removeEventListener("abort", onAbortRace);
              reject(new SIEConnectionError("Stream aborted by caller", "other"));
            };
            signal.addEventListener("abort", onAbortRace, { once: true });
            reader.read().then(
              (r) => {
                if (settled) return;
                settled = true;
                signal.removeEventListener("abort", onAbortRace);
                resolve(r);
              },
              (err) => {
                if (settled) return;
                settled = true;
                signal.removeEventListener("abort", onAbortRace);
                reject(err);
              },
            );
          });
        } else {
          result = await reader.read();
        }
      } catch (err) {
        if (err instanceof SIEConnectionError) throw err;
        if (signal?.aborted) {
          throw new SIEConnectionError("Stream aborted by caller", "other");
        }
        throw err;
      }

      if (result.done) {
        // Flush any trailing decoder state, then break out so the
        // post-loop tail handler can process a final event block that
        // arrived without a trailing blank line. The gateway normally
        // emits a `[DONE]` terminator, but a stream that ends with a
        // last `data:` line and no trailing `\n\n` would otherwise drop
        // that event (which can carry `finish_reason` / `usage`).
        buffer += decoder.decode();
        break;
      }

      buffer += decoder.decode(result.value, { stream: true });

      // Guard against an event frame that never terminates. The buffer only
      // retains bytes for an event we have not yet been able to split off;
      // if it grows past the cap, the peer is sending an unbounded frame and
      // we must fail rather than OOM the client.
      if (buffer.length > MAX_SSE_BUFFER_CHARS) {
        throw new SIEStreamError(
          `SSE event buffer exceeded ${MAX_SSE_BUFFER_CHARS} chars without an event terminator`,
        );
      }

      // Events are separated by a blank line (`\n\n`). We also accept
      // `\r\n\r\n` for compat with intermediaries that rewrite line
      // endings.
      let sepIdx: number;
      while (true) {
        const lfIdx = buffer.indexOf("\n\n");
        const crlfIdx = buffer.indexOf("\r\n\r\n");
        if (lfIdx === -1 && crlfIdx === -1) break;
        let sepLen = 2;
        if (lfIdx === -1) {
          sepIdx = crlfIdx;
          sepLen = 4;
        } else if (crlfIdx === -1) {
          sepIdx = lfIdx;
        } else {
          if (lfIdx < crlfIdx) {
            sepIdx = lfIdx;
          } else {
            sepIdx = crlfIdx;
            sepLen = 4;
          }
        }
        const eventBlock = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + sepLen);

        const payload = extractDataPayload(eventBlock);
        if (payload === null) continue;
        if (payload === SSE_DONE) {
          // Clean terminus: server signalled end-of-stream. No need to
          // cancel the body — it has already drained the relevant work.
          completedCleanly = true;
          if (completion) completion.done = true;
          return;
        }
        yield payload;
      }
    }

    // Stream ended (clean close). Flush a trailing event block that lacks
    // a terminating blank line — it goes through the same `extractDataPayload`
    // path and honours the `[DONE]` sentinel, so a final un-terminated
    // `data:` line (potentially carrying `finish_reason` / `usage`) is not
    // silently dropped. We strip a lone trailing newline first so a plain
    // `data: ...\n` tail (one newline, no blank line) is treated as a
    // complete single-line event.
    const tail = buffer.replace(/\r?\n$/, "");
    if (tail !== "") {
      const payload = extractDataPayload(tail);
      if (payload === SSE_DONE) {
        if (completion) completion.done = true;
      } else if (payload !== null) {
        yield payload;
      }
    }

    // Reaching here means we drained the stream to a graceful EOF (the
    // `result.done` break above flowed through the tail flush without a
    // `[DONE]`). That is a clean terminus.
    completedCleanly = true;
  } finally {
    if (signal) signal.removeEventListener("abort", onAbort);
    if (completedCleanly) {
      // Clean terminus — just drop our claim on the reader.
      try {
        reader.releaseLock();
      } catch {
        // Reader already released or cancelled.
      }
    } else {
      // Early / error teardown (caller break or early return, thrown
      // SIEStreamError, JSON parse RequestError, buffer-cap overflow, or an
      // abort). Cancel the reader so the underlying HTTP body/socket closes
      // and the worker stops generating. `cancel()` also releases the lock,
      // so we do not call `releaseLock()` afterwards. Swallow any rejection
      // (e.g. an already-released/cancelled reader) so we never surface an
      // unhandled rejection from the teardown path.
      await reader.cancel().catch(() => {
        // Reader already released or cancelled, or the source's cancel
        // algorithm threw — nothing we can do at teardown.
      });
    }
  }
}

/**
 * Pull the `data:` payload out of a single SSE event block.
 *
 * Returns `null` for events with no `data:` line (keep-alive comments,
 * `event:`-only frames). When multiple `data:` lines are present they
 * are joined with `\n` per the SSE spec, though the SIE gateway never
 * emits multi-line payloads today.
 */
function extractDataPayload(block: string): string | null {
  const lines = block.split(/\r?\n/);
  const parts: string[] = [];
  for (const line of lines) {
    if (line === "" || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      // Per spec, a single leading space after the colon is stripped.
      let value = line.slice(5);
      if (value.startsWith(" ")) value = value.slice(1);
      parts.push(value);
    }
    // Other field names (`event:`, `id:`, `retry:`) are ignored — the
    // SIE gateway does not emit them.
  }
  if (parts.length === 0) return null;
  return parts.join("\n");
}
