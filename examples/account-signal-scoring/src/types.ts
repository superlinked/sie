export type SignalDirection = "risk" | "opportunity";
export type SignalSeverity = "major" | "medium" | "minor";

export type SignalType =
  // risk
  | "stripe_cancellation"
  | "usage_drop"
  | "negative_support_ticket"
  // opportunity
  | "usage_near_limit"
  | "renewal_approaching"
  | "positive_support_ticket";

export interface Signal {
  type: SignalType;
  /** Free-form context surfaced in the brief and to the extractor. */
  note?: string;
  /** When it was detected (display string). */
  detected?: string;
  /** Optional magnitude, e.g. usage-drop percentage (0-100). */
  value?: number | null;
}

export interface Contact {
  name: string;
  title: string;
}

export interface Account {
  id: string;
  name: string;
  domain: string;
  owner: string;
  arr: number;
  seats: number;
  seatsUsed: number;
  renewalDays: number;
  contact: Contact;
  signals: Signal[];
}

/** A past-outcome pattern the live account is ranked against. */
export interface Playbook {
  id: string;
  direction: SignalDirection;
  label: string;
  summary: string;
  outcome: string;
  play: string;
}

export interface IndexRecord {
  id: string;
  vector: number[];
}

/** The account brief written back to the CRM / triage board. */
export interface AccountBrief {
  summary: string;
  drivers: string;
  recommendedPlay: string;
  arrAtStake: number | null;
  /** "sie-chat" when drafted by the LLM, "deterministic" otherwise. */
  source: string;
}
