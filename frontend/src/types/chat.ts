import type { QuerySource } from "./api";

export interface ConversationItem {
  readonly id: string;
  readonly question: string;
  readonly answer: string;
  readonly sources: QuerySource[];
}
