// Typed SSE events streamed to the browser during a scoring run.

import type { ScoreBand } from "./signals.js";
import type { SignalDirection } from "./types.js";

export interface ExtractedEntity {
  label: string;
  text: string;
  score: number;
}

export interface PlaybookHit {
  id: string;
  label: string;
  direction: SignalDirection;
  summary: string;
  outcome: string;
  play: string;
  score: number;
}

export interface BriefData {
  summary: string;
  drivers: string;
  recommendedPlay: string;
  arrAtStake: number | null;
  source: string;
}

export type ScoreEvent =
  | { type: "models"; data: { encoder: string; reranker: string; extractor: string; chat: string } }
  | {
      type: "signals";
      data: { score: number; band: ScoreBand; direction: SignalDirection; reason: string };
    }
  | { type: "extracting" }
  | { type: "extracted"; data: { entities: ExtractedEntity[]; ms: number } }
  | { type: "encoding" }
  | { type: "encoded"; data: { dim: number; ms: number } }
  | { type: "scoring" }
  | { type: "scored"; data: { hits: PlaybookHit[]; topPlaybook: string; ms: number } }
  | { type: "briefing" }
  | { type: "brief"; data: BriefData & { ms: number } }
  | { type: "done"; data: { totalMs: number } }
  | { type: "error"; data: { stage: string; message: string } };
