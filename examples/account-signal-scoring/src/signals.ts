// Turn a pile of account signals into one score a team can act on.
//
// This is the deterministic, auditable half of the pipeline: each signal has a
// direction (risk vs. opportunity) and a severity weight, and the weights roll
// up into a single 0-100 score plus a red/amber/green band. SIE then does the
// *ranking* — matching the account's story against a corpus of past outcomes to
// pick the play (see score.ts).

import type { Account, Signal, SignalDirection, SignalSeverity, SignalType } from "./types.js";

/** Numeric weight per severity. */
export const SEVERITY_WEIGHT: Record<SignalSeverity, number> = {
  major: 10,
  medium: 5,
  minor: 2,
};

/** Default direction + severity for each known signal type. */
export const SIGNAL_CATALOG: Record<
  SignalType,
  { direction: SignalDirection; severity: SignalSeverity; label: string }
> = {
  stripe_cancellation: { direction: "risk", severity: "major", label: "Stripe cancellation / downgrade" },
  usage_drop: { direction: "risk", severity: "medium", label: "Usage drop" },
  negative_support_ticket: { direction: "risk", severity: "minor", label: "Negative support ticket" },
  usage_near_limit: { direction: "opportunity", severity: "major", label: "Usage near limit" },
  renewal_approaching: { direction: "opportunity", severity: "medium", label: "Renewal approaching" },
  positive_support_ticket: { direction: "opportunity", severity: "minor", label: "Positive support ticket" },
};

export type ScoreBand = "red" | "amber" | "green";

export interface AccountScore {
  /** 0-100. Higher = more urgent (churn risk or expansion pull). */
  score: number;
  band: ScoreBand;
  /** Net lean of the signal set. */
  direction: SignalDirection;
  /** Human-readable reason string built from the active signals. */
  reason: string;
}

const clamp = (n: number) => Math.max(0, Math.min(100, Math.round(n)));

/** Map a 0-100 score to a band. >40 red, 10-40 amber, <10 green. */
export function bandForScore(score: number): ScoreBand {
  if (score > 40) return "red";
  if (score >= 10) return "amber";
  return "green";
}

/**
 * Roll a set of signals up into one score.
 *
 * Stripe cancellation is a hard short-circuit to the top of the risk board
 * (an active cancellation is unambiguous). Otherwise risk signals add and
 * opportunity signals subtract, each weighted by severity; a usage_drop is
 * additionally scaled by its magnitude when provided.
 */
export function scoreSignals(signals: Signal[]): AccountScore {
  if (signals.some((s) => s.type === "stripe_cancellation")) {
    return {
      score: 100,
      band: "red",
      direction: "risk",
      reason: "Stripe subscription cancellation / downgrade staged",
    };
  }

  let risk = 0;
  let opportunity = 0;
  const reasons: string[] = [];

  for (const s of signals) {
    const meta = SIGNAL_CATALOG[s.type];
    if (!meta) continue;
    let weight = SEVERITY_WEIGHT[meta.severity];
    // Scale a usage drop by how bad it is (a 41% drop should sting more).
    if (s.type === "usage_drop" && typeof s.value === "number") {
      weight = Math.max(weight, s.value * 0.6);
    }
    if (meta.direction === "risk") risk += weight;
    else opportunity += weight;
    if (s.note) reasons.push(s.note);
    else reasons.push(meta.label);
  }

  const direction: SignalDirection = risk >= opportunity ? "risk" : "opportunity";
  const score = clamp(direction === "risk" ? risk : opportunity);
  return {
    score,
    band: bandForScore(score),
    direction,
    reason: reasons.join("; ") || "No active signals",
  };
}

/**
 * Build the natural-language account context that gets embedded and passed to
 * the extractor and the LLM. Keeping this as prose (not a feature vector) is
 * what lets a small dense encoder match it against the outcome playbooks.
 */
export function buildAccountSummary(account: Account): string {
  const { score, direction } = scoreSignals(account.signals);
  const seatPct = Math.round((account.seatsUsed / account.seats) * 100);
  const signalText = account.signals
    .map((s) => `${SIGNAL_CATALOG[s.type]?.label ?? s.type}${s.note ? ` (${s.note})` : ""}`)
    .join("; ");
  return [
    `${account.name} (${account.domain}) is a ${direction} account with a signal score of ${score}.`,
    `ARR $${account.arr.toLocaleString()}, ${account.seatsUsed}/${account.seats} seats used (${seatPct}%), renewal in ${account.renewalDays} days.`,
    `Primary contact: ${account.contact.name}, ${account.contact.title}. Account owner: ${account.owner}.`,
    `Active signals: ${signalText || "none"}.`,
  ].join(" ");
}

/** ARR exposed by the account (used as "at stake" in the brief). */
export function arrAtStake(account: Account): number | null {
  return account.arr ?? null;
}
