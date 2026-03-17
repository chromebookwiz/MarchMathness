import type { Player, TeamRoster } from './types';

// ── ESPN team ID mapping for every tournament team ─────────────────────────
export const ESPN_IDS: Record<string, number> = {
  // East
  'duke': 150,
  'alabama': 333,
  'wisconsin': 275,
  'arizona': 12,
  'oregon': 2483,
  'byu': 252,
  'saint-marys': 2608,
  'miss-state': 344,
  'baylor': 239,
  'vanderbilt': 238,
  'vcu': 2670,
  'liberty': 2335,
  'akron': 2006,
  'montana': 149,
  'robert-morris': 2523,
  'longwood': 2393,
  // South
  'auburn': 2,
  'michigan-state': 127,
  'iowa-state': 66,
  'texas-am': 245,
  'michigan': 130,
  'mississippi': 145,
  'gonzaga': 2250,
  'louisville': 97,
  'creighton': 156,
  'new-mexico': 167,
  'drake': 2181,
  'uc-san-diego': 2706,
  'yale': 43,
  'lipscomb': 2348,
  'bryant': 2803,
  'winthrop': 2679,
  // Midwest
  'tennessee': 2633,
  'st-johns': 2599,
  'kentucky': 96,
  'maryland': 120,
  'texas': 251,
  'kansas': 2305,
  'ucla': 26,
  'purdue': 2509,
  'illinois': 356,
  'utah-state': 254,
  'nc-state': 152,
  'colorado-state': 36,
  'grand-canyon': 2253,
  'troy': 2653,
  'omaha': 2437,
  'sfa': 2540,
  // West
  'houston': 248,
  'florida': 57,
  'texas-tech': 2641,
  'marquette': 269,
  'clemson': 228,
  'pittsburgh': 221,
  'dayton': 2196,
  'oklahoma': 201,
  'arkansas': 8,
  'notre-dame': 87,
  'mcneese': 2403,
  'colorado': 38,
  'unc-asheville': 2462,
  'wofford': 2780,
  'norfolk-state': 2450,
  'howard': 47,
};

const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours
const CACHE_PREFIX = 'mmn_roster_v2_';

// ── localStorage caching helpers ──────────────────────────────────────────

function cacheGet(teamId: string): Player[] | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + teamId);
    if (!raw) return null;
    const { players, ts } = JSON.parse(raw) as { players: Player[]; ts: number };
    if (Date.now() - ts > CACHE_TTL_MS) {
      localStorage.removeItem(CACHE_PREFIX + teamId);
      return null;
    }
    return players;
  } catch {
    return null;
  }
}

function cacheSet(teamId: string, players: Player[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(CACHE_PREFIX + teamId, JSON.stringify({ players, ts: Date.now() }));
  } catch {
    // Ignore storage quota errors
  }
}

// ── ESPN fetch with retry ─────────────────────────────────────────────────

async function espnFetch(endpoint: string, retries = 3): Promise<unknown> {
  const url = `/api/espn?endpoint=${encodeURIComponent(endpoint)}`;
  let lastErr: unknown;

  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      lastErr = err;
      if (attempt < retries - 1) {
        await new Promise(r => setTimeout(r, 400 * Math.pow(2, attempt)));
      }
    }
  }
  throw lastErr;
}

// ── Parse ESPN roster response into Player[] ──────────────────────────────

function parseAthleteStats(athlete: Record<string, unknown>): Partial<Player> {
  const stats = (athlete.statistics as Record<string, unknown> | undefined);
  const splits = (stats?.splits as Record<string, unknown> | undefined);
  const categories = (splits?.categories as Array<Record<string, unknown>> | undefined) ?? [];

  const statMap: Record<string, number> = {};
  for (const cat of categories) {
    const statArr = (cat.stats as Array<Record<string, unknown>> | undefined) ?? [];
    for (const s of statArr) {
      if (typeof s.name === 'string' && typeof s.value === 'number') {
        statMap[s.name] = s.value;
      }
    }
  }

  const fga   = statMap['avgFieldGoalsAttempted']            ?? statMap['FGA']  ?? 8;
  const fg3a  = statMap['avgThreePointFieldGoalsAttempted']  ?? statMap['3PA']  ?? 2;
  const fta   = statMap['avgFreeThrowsAttempted']            ?? statMap['FTA']  ?? 2;
  const fg    = statMap['avgFieldGoalsMade']                 ?? statMap['FGM']  ?? 3.5;
  const fg3   = statMap['avgThreePointFieldGoalsMade']       ?? statMap['3PM']  ?? 0.8;
  const ft    = statMap['avgFreeThrowsMade']                 ?? statMap['FTM']  ?? 1.5;
  const ppg   = statMap['avgPoints']   ?? statMap['PTS']  ?? fg * 2 + fg3 + ft;
  const rpg   = statMap['avgRebounds'] ?? statMap['REB']  ?? 3.5;
  const apg   = statMap['avgAssists']  ?? statMap['AST']  ?? 1.5;
  const spg   = statMap['avgSteals']   ?? statMap['STL']  ?? 0.6;
  const bpg   = statMap['avgBlocks']   ?? statMap['BLK']  ?? 0.4;
  const topg  = statMap['avgTurnovers']?? statMap['TOV']  ?? 1.2;
  const fpg   = statMap['avgFouls']    ?? statMap['PF']   ?? 2.2;
  const mpg   = statMap['avgMinutes']  ?? statMap['MIN']  ?? 22;

  const fgPct  = fga  > 0 ? fg  / fga  : 0.42;
  const fg3Pct = fg3a > 0 ? fg3 / fg3a : 0.33;
  const ftPct  = fta  > 0 ? ft  / fta  : 0.72;

  const ts = ppg / (2 * (fga + 0.44 * fta + 0.001));
  const usage = Math.min(0.40, (fga + 0.44 * fta + topg) / 28);

  const obpm = (ppg - 10) * 0.25 + apg * 0.5 - topg * 0.8 + (fg3 - 0.8) * 0.4;
  const dbpm = spg * 1.5 + bpg * 1.2 - fpg * 0.2;
  const bpm  = obpm + dbpm;

  return {
    mpg, ppg, rpg, apg, spg, bpg, topg, fpg,
    fgPct, fg3Pct, ftPct,
    fgaPerGame: fga, fg3aPerGame: fg3a, ftaPerGame: fta,
    usageRate: usage,
    trueShootingPct: Math.max(0, Math.min(1, ts)),
    offRtg: 100 + ppg * 1.2 + apg * 0.8 - topg * 1.5,
    defRtg: 100 + bpg * 1.5 + spg * 1.2 - fpg * 0.3,
    obpm, dbpm, bpm,
    starScore: Math.min(100, Math.max(0,
      ppg * 2 + rpg * 1.2 + apg * 1.5 + bpm * 3 + (ts - 0.5) * 30
    )),
  };
}

// ── Generate synthetic fallback players from team averages ─────────────────

function generateFallbackRoster(teamId: string): Player[] {
  // Produce a realistic-looking 10-player roster when ESPN fails
  const positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'G', 'F', 'F', 'C'];
  const years = ['JR', 'SR', 'SO', 'JR', 'SR', 'FR', 'SO', 'JR', 'SR', 'FR'];
  const mpgs  = [32, 30, 29, 27, 26, 18, 16, 15, 13, 10];

  // Seed the RNG deterministically from teamId so the same team always gets the same fallback
  let seed = teamId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };

  return positions.map((pos, i) => {
    const isStarter = i < 5;
    const mpg = mpgs[i];
    const ppg = isStarter ? 10 + rand() * 10 : 4 + rand() * 7;
    const rpg = pos === 'C' || pos === 'PF' ? 5 + rand() * 5 : 2 + rand() * 4;
    const apg = pos === 'PG' ? 3 + rand() * 4 : 0.5 + rand() * 2.5;
    const spg = 0.3 + rand() * 1.2;
    const bpg = pos === 'C' ? 0.5 + rand() * 1.5 : 0.1 + rand() * 0.6;
    const topg = 1 + rand() * 2;
    const fgPct = 0.40 + rand() * 0.12;
    const ftPct = 0.65 + rand() * 0.25;
    const ts = 0.50 + rand() * 0.12;
    const usage = isStarter ? 0.18 + rand() * 0.10 : 0.08 + rand() * 0.08;
    const bpm = isStarter ? -2 + rand() * 6 : -4 + rand() * 3;

    return {
      id: `${teamId}_p${i}`,
      name: `Player ${i + 1}`,
      jersey: String(Math.floor(rand() * 35)),
      position: pos,
      year: years[i],
      height: `${5 + (pos === 'C' || pos === 'PF' ? 1 : 0)}'${Math.floor(rand() * 12)}"`,
      weight: 180 + Math.floor(rand() * 60),
      mpg, ppg, rpg, apg, spg, bpg, topg, fpg: 1.5 + rand() * 1.5,
      fgPct, fg3Pct: 0.30 + rand() * 0.15, ftPct,
      fgaPerGame: ppg / (2 * fgPct + 0.001),
      fg3aPerGame: 2 + rand() * 4,
      ftaPerGame: 1.5 + rand() * 3,
      usageRate: usage,
      trueShootingPct: ts,
      offRtg: 100 + ppg * 1.2 + apg * 0.8 - topg * 1.5,
      defRtg: 100 + bpg * 1.5 + spg * 1.2,
      bpm, obpm: bpm * 0.65, dbpm: bpm * 0.35,
      isStarter,
      starScore: Math.min(100, Math.max(0, ppg * 2 + rpg * 1.2 + apg * 1.5 + bpm * 3 + (ts - 0.5) * 30)),
    } as Player;
  });
}

function positionFromEspn(pos: Record<string, unknown> | null | undefined): string {
  if (!pos) return 'G';
  const abbr = (pos.abbreviation as string | undefined)?.toUpperCase() ?? '';
  if (abbr.includes('PG')) return 'PG';
  if (abbr.includes('SG')) return 'SG';
  if (abbr.includes('SF')) return 'SF';
  if (abbr.includes('PF')) return 'PF';
  if (abbr.includes('C'))  return 'C';
  if (abbr.includes('G'))  return 'G';
  if (abbr.includes('F'))  return 'F';
  return 'G';
}

function expFromYears(years: number): string {
  return ['FR', 'SO', 'JR', 'SR'][Math.min(years, 3)] ?? 'SR';
}

export async function fetchTeamRoster(teamId: string): Promise<Player[]> {
  const espnId = ESPN_IDS[teamId];
  if (!espnId) return generateFallbackRoster(teamId);

  // Check cache first
  const cached = cacheGet(teamId);
  if (cached) return cached;

  try {
    const data = await espnFetch(`teams/${espnId}/roster?enable=stats`) as Record<string, unknown>;
    const athletes = (data.athletes as Array<Record<string, unknown>> | undefined) ?? [];

    if (!athletes.length) {
      const fallback = generateFallbackRoster(teamId);
      cacheSet(teamId, fallback);
      return fallback;
    }

    const players: Player[] = athletes
      .filter(a => {
        const s = a.status as Record<string, unknown> | undefined;
        return s?.type !== 'inactive';
      })
      .map(a => {
        const pos  = a.position as Record<string, unknown> | undefined;
        const exp  = a.experience as Record<string, unknown> | undefined;
        const stats = parseAthleteStats(a);
        return {
          id:       String(a.id ?? Math.random()),
          name:     String(a.displayName ?? a.fullName ?? 'Unknown'),
          jersey:   String(a.jersey ?? '0'),
          position: positionFromEspn(pos ?? null),
          year:     expFromYears((exp?.years as number | undefined) ?? 0),
          height:   String(a.displayHeight ?? '6\'0"'),
          weight:   Number(a.weight ?? 190),
          isStarter: false,
          ...stats,
        } as Player;
      })
      .sort((a, b) => b.mpg - a.mpg);

    // Mark starters
    players.slice(0, 5).forEach(p => { p.isStarter = true; });

    // Normalize usage rates to sum to ~1
    const totalUsage = players.reduce((s, p) => s + p.usageRate, 0);
    if (totalUsage > 0) players.forEach(p => { p.usageRate = p.usageRate / totalUsage; });

    cacheSet(teamId, players);
    return players;
  } catch {
    // On failure: return fallback (but don't cache — try again next time)
    return generateFallbackRoster(teamId);
  }
}

// ── Compute team-level features from player roster ────────────────────────

export function computeRosterMetrics(players: Player[]): TeamRoster {
  if (!players.length) {
    return {
      teamId: '',
      players: [],
      fetchedAt: Date.now(),
      starPlayerBPM: 0,
      depthScore: 0.5,
      starReliance: 0.25,
      avgExperienceYears: 0.5,
      recruitingRank: 50,
    };
  }

  const sorted = [...players].sort((a, b) => b.bpm - a.bpm);
  const starBPM = sorted[0]?.bpm ?? 0;

  const top8 = sorted.slice(0, 8);
  const depthScore = top8.length >= 6
    ? (top8[4]?.ppg ?? 0) > 0 ? (top8[5]?.ppg ?? 0) / (top8[4]?.ppg ?? 1) : 0.5
    : 0.3;

  const totalPpg = players.reduce((s, p) => s + p.ppg, 0);
  const starReliance = totalPpg > 0 ? (sorted[0]?.ppg ?? 0) / totalPpg : 0.25;

  const yearNums = players.map(p => {
    const m: Record<string, number> = { FR: 1, SO: 2, JR: 3, SR: 4, Grad: 4.5 };
    return m[p.year] ?? 2;
  });
  const avgExp = yearNums.reduce((a, b) => a + b, 0) / (yearNums.length || 1);
  const avgExpNorm = (avgExp - 1) / 3.5;

  return {
    teamId: '',
    players,
    fetchedAt: Date.now(),
    starPlayerBPM: starBPM,
    depthScore: Math.min(1, Math.max(0, depthScore)),
    starReliance,
    avgExperienceYears: avgExpNorm,
    recruitingRank: 50,
  };
}

// ── Fetch rosters for all 64 teams concurrently ────────────────────────────

export async function fetchAllRosters(
  teamIds: string[],
  onProgress: (done: number, total: number) => void,
): Promise<Map<string, Player[]>> {
  const result = new Map<string, Player[]>();
  let done = 0;

  const BATCH = 6;
  for (let i = 0; i < teamIds.length; i += BATCH) {
    const batch = teamIds.slice(i, i + BATCH);
    const fetched = await Promise.all(
      batch.map(async id => {
        const players = await fetchTeamRoster(id);
        return { id, players };
      })
    );
    for (const { id, players } of fetched) {
      result.set(id, players);
      done++;
      onProgress(done, teamIds.length);
    }
    await new Promise(r => setTimeout(r, 30));
  }

  return result;
}
