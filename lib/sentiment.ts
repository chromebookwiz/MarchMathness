import type { Team } from './types';

const CACHE_TTL_MS = 3 * 60 * 60 * 1000; // 3 hours
const CACHE_PREFIX = 'mmn_sentiment_v1_';

function cacheGet(teamId: string): number | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(`${CACHE_PREFIX}${teamId}`);
    if (!raw) return null;
    const { score, ts } = JSON.parse(raw) as { score: number; ts: number };
    if (Date.now() - ts > CACHE_TTL_MS) {
      localStorage.removeItem(`${CACHE_PREFIX}${teamId}`);
      return null;
    }
    return score;
  } catch {
    return null;
  }
}

function cacheSet(teamId: string, score: number): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(`${CACHE_PREFIX}${teamId}`, JSON.stringify({ score, ts: Date.now() }));
  } catch {
    // ignore quota
  }
}

function scoreText(text: string): number {
  const pos = ['win', 'dominate', 'upset', 'outscore', 'clutch', 'victory', 'lead', 'hot', 'streak', 'beat', 'crush'];
  const neg = ['injury', 'struggle', 'loss', 'slump', 'drop', 'hurt', 'doubt', 'weird', 'worry', 'scuffle'];
  const normalized = text.toLowerCase();
  let score = 0;
  for (const w of pos) if (normalized.includes(w)) score += 1;
  for (const w of neg) if (normalized.includes(w)) score -= 1;
  return score;
}

async function fetchTeamHeadlines(espnId: number): Promise<string[]> {
  try {
    const res = await fetch(`/api/sentiment?teamId=${espnId}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { headlines: string[] };
    return Array.isArray(data.headlines) ? data.headlines : [];
  } catch {
    return [];
  }
}

export async function fetchSentimentScores(teams: Team[]): Promise<Map<string, number>> {
  const map = new Map<string, number>();

  await Promise.all(teams.map(async team => {
    const cached = cacheGet(team.id);
    if (cached != null) {
      map.set(team.id, cached);
      return;
    }

    const headlines = await fetchTeamHeadlines(team.espnId);
    if (headlines.length === 0) {
      map.set(team.id, 0);
      cacheSet(team.id, 0);
      return;
    }

    const score = headlines.reduce((s, h) => s + scoreText(h), 0) / headlines.length;
    const clipped = Math.max(-1, Math.min(1, score / 2));
    map.set(team.id, clipped);
    cacheSet(team.id, clipped);
  }));

  return map;
}
