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

async function espnFetch(endpoint: string): Promise<unknown> {
  const url = `/api/espn?endpoint=${encodeURIComponent(endpoint)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ESPN fetch failed: ${endpoint}`);
  return res.json();
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

  const fga   = statMap['avgFieldGoalsAttempted']   ?? statMap['FGA']   ?? 8;
  const fg3a  = statMap['avgThreePointFieldGoalsAttempted'] ?? statMap['3PA'] ?? 2;
  const fta   = statMap['avgFreeThrowsAttempted']   ?? statMap['FTA']   ?? 2;
  const fg    = statMap['avgFieldGoalsMade']         ?? statMap['FGM']   ?? 3.5;
  const fg3   = statMap['avgThreePointFieldGoalsMade'] ?? statMap['3PM'] ?? 0.8;
  const ft    = statMap['avgFreeThrowsMade']         ?? statMap['FTM']   ?? 1.5;
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

  // True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
  const ts = ppg / (2 * (fga + 0.44 * fta + 0.001));

  // Approximate usage (will be recalculated relative to team)
  const usage = Math.min(0.40, (fga + 0.44 * fta + topg) / 28);

  // BPM approximation
  const obpm = (ppg - 10) * 0.25 + apg * 0.5 - topg * 0.8 + (fg3 - 0.8) * 0.4;
  const dbpm = spg * 1.5 + bpg * 1.2 - fpg * 0.2;
  const bpm  = obpm + dbpm;

  return {
    mpg, ppg, rpg, apg, spg, bpg, topg, fpg,
    fgPct, fg3Pct, ftPct,
    fgaPerGame: fga, fg3aPerGame: fg3a, ftaPerGame: fta,
    usageRate: usage,
    trueShootingPct: ts,
    offRtg: 100 + ppg * 1.2 + apg * 0.8 - topg * 1.5,
    defRtg: 100 + bpg * 1.5 + spg * 1.2 - fpg * 0.3,
    obpm, dbpm, bpm,
    starScore: Math.min(100, Math.max(0,
      ppg * 2 + rpg * 1.2 + apg * 1.5 + bpm * 3 + (ts - 0.5) * 30
    )),
  };
}

function positionFromEspn(pos: Record<string, unknown> | null | undefined): string {
  if (!pos) return 'G';
  const abbr = (pos.abbreviation as string | undefined)?.toUpperCase() ?? '';
  if (abbr.includes('PG') || abbr === 'G') return 'PG';
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
  if (!espnId) return [];

  try {
    const data = await espnFetch(`teams/${espnId}/roster?enable=stats`) as Record<string, unknown>;
    const athletes = (data.athletes as Array<Record<string, unknown>> | undefined) ?? [];

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
      .sort((a, b) => b.mpg - a.mpg); // sort by minutes

    // Mark starters
    players.slice(0, 5).forEach(p => { p.isStarter = true; });

    // Normalize usage rates to sum to ~1
    const totalUsage = players.reduce((s, p) => s + p.usageRate, 0);
    if (totalUsage > 0) {
      players.forEach(p => { p.usageRate = p.usageRate / totalUsage; });
    }

    return players;
  } catch {
    return [];
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

  // Depth: how close the 6th player is to the 5th
  const top8 = sorted.slice(0, 8);
  const depthScore = top8.length >= 6
    ? (top8[4]?.ppg ?? 0) > 0 ? (top8[5]?.ppg ?? 0) / (top8[4]?.ppg ?? 1) : 0.5
    : 0.3;

  // Star reliance: how much of scoring comes from top player
  const totalPpg = players.reduce((s, p) => s + p.ppg, 0);
  const starReliance = totalPpg > 0 ? (sorted[0]?.ppg ?? 0) / totalPpg : 0.25;

  // Experience: convert year strings to numbers
  const yearNums = players.map(p => {
    const m: Record<string, number> = { FR: 1, SO: 2, JR: 3, SR: 4, Grad: 4.5 };
    return m[p.year] ?? 2;
  });
  const avgExp = yearNums.reduce((a, b) => a + b, 0) / (yearNums.length || 1);
  const avgExpNorm = (avgExp - 1) / 3.5; // 0-1

  return {
    teamId: '',
    players,
    fetchedAt: Date.now(),
    starPlayerBPM: starBPM,
    depthScore: Math.min(1, Math.max(0, depthScore)),
    starReliance,
    avgExperienceYears: avgExpNorm,
    recruitingRank: 50, // placeholder
  };
}

// ── Fetch rosters for all 64 teams concurrently ────────────────────────────

export async function fetchAllRosters(
  teamIds: string[],
  onProgress: (done: number, total: number) => void,
): Promise<Map<string, Player[]>> {
  const result = new Map<string, Player[]>();
  let done = 0;

  // Fetch in batches of 6 to avoid overwhelming the proxy
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
    // Small yield between batches
    await new Promise(r => setTimeout(r, 50));
  }

  return result;
}
