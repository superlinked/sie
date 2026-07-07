// Account brief generation.
//
// If SIE_CHAT_MODEL is set, SIE drafts the brief through the OpenAI-compatible
// /v1/chat/completions endpoint, grounded in the top-matched playbook. If it is
// not set (the CPU-only default) or the call fails, we fall back to a
// deterministic brief so the triage board is always populated. This mirrors the
// original Attio project, which raced the LLM against a deterministic writer.
//
// The published SIE SDK (0.3.x) exposes encode/score/extract; chat/completions
// is the OpenAI-compatible HTTP surface, so we call it directly with fetch —
// exactly as the original project did against the SIE gateway.

import { config } from "./config.js";
import { arrAtStake, scoreSignals } from "./signals.js";
import type { AccountBrief, Account, Playbook } from "./types.js";

const money = (n: number | null) => (n == null ? "n/a" : `$${Math.round(n).toLocaleString()}`);

/** Whether the LLM brief path is configured. */
export function chatEnabled(): boolean {
  return Boolean(config.chatModel);
}

const SYSTEM_PROMPT =
  "You are the Head of Customer Success. Given a customer account context and the " +
  "closest-matching historical playbook, produce a concise account brief. Respond " +
  'ONLY with JSON: {"summary": string, "drivers": string, "recommendedPlay": string}.';

/** Deterministic brief assembled from the account's own data. */
export function deterministicBrief(account: Account, topPlaybook?: Playbook): AccountBrief {
  const s = scoreSignals(account.signals);
  const lead = s.direction === "risk" ? "at risk" : "an expansion opportunity";
  return {
    summary:
      `${account.name} is ${lead} (${s.band.toUpperCase()}, signal score ${Math.round(s.score)}). ` +
      `${s.reason}. ARR ${money(account.arr)}, renewal in ${account.renewalDays} days.`,
    drivers: s.reason,
    recommendedPlay: topPlaybook?.play ?? "Review the account and decide on the next play.",
    arrAtStake: arrAtStake(account),
    source: "deterministic",
  };
}

interface ChatResponse {
  choices?: { message?: { content?: string } }[];
}

/**
 * Draft the brief via SIE chat/completions, grounded in the matched playbook.
 * Falls back to the deterministic brief on any error.
 */
export async function generateBrief(
  account: Account,
  topPlaybook: Playbook | undefined,
  summary: string,
): Promise<AccountBrief> {
  if (!chatEnabled()) return deterministicBrief(account, topPlaybook);

  const userContent = [
    `Account context:\n${summary}`,
    topPlaybook
      ? `Closest historical playbook: ${topPlaybook.label} — ${topPlaybook.summary} Recommended play: ${topPlaybook.play}`
      : "No close historical playbook matched.",
  ].join("\n\n");

  try {
    const res = await fetch(`${config.sieUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(config.sieApiKey ? { Authorization: `Bearer ${config.sieApiKey}` } : {}),
      },
      body: JSON.stringify({
        model: config.chatModel,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userContent },
        ],
        response_format: { type: "json_object" },
        max_tokens: 512,
      }),
      signal: AbortSignal.timeout(120_000),
    });
    if (!res.ok) throw new Error(`chat/completions ${res.status}: ${await res.text()}`);
    const json = (await res.json()) as ChatResponse;
    const content = json.choices?.[0]?.message?.content;
    if (typeof content !== "string" || content.length === 0) {
      throw new Error("chat/completions returned no content");
    }
    const parsed = JSON.parse(content) as Partial<AccountBrief>;
    return {
      summary: parsed.summary ?? "",
      drivers: parsed.drivers ?? "",
      recommendedPlay: parsed.recommendedPlay ?? topPlaybook?.play ?? "",
      arrAtStake: arrAtStake(account),
      source: "sie-chat",
    };
  } catch {
    return deterministicBrief(account, topPlaybook);
  }
}
