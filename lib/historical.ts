import type { Team } from './types';
import { computeFeatures } from './simulation';

export interface GameRecord {
  id: number;
  season: number;
  date: string;
  home_team: { id: number; name: string; abbreviation: string; full_name?: string };
  visitor_team: { id: number; name: string; abbreviation: string; full_name?: string };
  home_team_score: number;
  visitor_team_score: number;
}

export interface TeamSeasonStats {
  teamId: number;
  season: number;
  teamName: string;
  games: number;
  wins: number;
  losses: number;
  pointsFor: number;
  pointsAgainst: number;
  pace: number;
}

const API_BASE = 'https://www.balldontlie.io/api/v1';

async function fetchAllGamesForSeason(season: number, maxPages = 20): Promise<GameRecord[]> {
  const games: GameRecord[] = [];
  let page = 1;

  while (true) {
    const res = await fetch(`/api/balldontlie/games?season=${season}&page=${page}`);
    if (!res.ok) break;
    const body = await res.json();
    const data = body.data as GameRecord[];
    if (!data?.length) break;
    games.push(...data);

    const totalPages = body.meta?.total_pages ?? page;
    if (page >= totalPages || page >= maxPages) break;
    page += 1;
  }

  return games;
}

export async function fetchHistoricalSeasonStats(seasons: number[]): Promise<TeamSeasonStats[]> {
  const allStats: TeamSeasonStats[] = [];
  const maxPages = Number(process.env.HISTORICAL_MAX_PAGES ?? 20);

  for (const season of seasons) {
    const games = await fetchAllGamesForSeason(season, maxPages);
    const statsMap = new Map<number, TeamSeasonStats>();

    for (const g of games) {
      const homeId = g.home_team.id;
      const awayId = g.visitor_team.id;

      const ensure = (id: number, name: string) => {
        if (!statsMap.has(id)) {
          statsMap.set(id, {
            teamId: id,
            season,
            teamName: name,
            games: 0,
            wins: 0,
            losses: 0,
            pointsFor: 0,
            pointsAgainst: 0,
            pace: 70,
          });
        }
        return statsMap.get(id)!;
      };

      const homeName = g.home_team.full_name ?? g.home_team.name;
      const awayName = g.visitor_team.full_name ?? g.visitor_team.name;
      const home = ensure(homeId, homeName);
      const away = ensure(awayId, awayName);

      const homePts = g.home_team_score;
      const awayPts = g.visitor_team_score;

      home.games += 1;
      away.games += 1;
      home.pointsFor += homePts;
      home.pointsAgainst += awayPts;
      away.pointsFor += awayPts;
      away.pointsAgainst += homePts;

      if (homePts > awayPts) {
        home.wins += 1;
        away.losses += 1;
      } else {
        away.wins += 1;
        home.losses += 1;
      }
    }

    allStats.push(...statsMap.values());
  }

  return allStats;
}

export interface GameOutcome {
  season: number;
  date: string;
  teamA: string;
  teamB: string;
  winner: string;
}

export async function fetchHistoricalOutcomes(seasons: number[]): Promise<GameOutcome[]> {
  const output: GameOutcome[] = [];
  const maxPages = Number(process.env.HISTORICAL_MAX_PAGES ?? 20);
  for (const season of seasons) {
    const games = await fetchAllGamesForSeason(season, maxPages);
    for (const g of games) {
      const homeName = g.home_team.full_name || g.home_team.name;
      const awayName = g.visitor_team.full_name || g.visitor_team.name;
      const winner = g.home_team_score > g.visitor_team_score ? homeName : awayName;
      output.push({ season, date: g.date, teamA: homeName, teamB: awayName, winner });
    }
  }
  return output;
}

export function mapHistoricalStatsToTeams(teams: Team[], stats: TeamSeasonStats[]): Map<string, Team> {
  const out = new Map<string, Team>();

  const normalize = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
  const teamsByNorm = new Map<string, Team>();
  teams.forEach(t => {
    teamsByNorm.set(normalize(t.name), t);
    if (t.abbreviation) teamsByNorm.set(normalize(t.abbreviation), t);
  });

  // Group stats by season to compute seeds/ranks within each year
  const statsBySeason = new Map<number, TeamSeasonStats[]>();
  for (const stat of stats) {
    const arr = statsBySeason.get(stat.season) ?? [];
    arr.push(stat);
    statsBySeason.set(stat.season, arr);
  }

  for (const [season, seasonStats] of statsBySeason) {
    // Compute net rating (adjOE - adjDE) for sorting
    const seasonRanking = seasonStats
      .map(s => ({ ...s, netRating: (s.pointsFor - s.pointsAgainst) / Math.max(1, s.games) }))
      .sort((a, b) => b.netRating - a.netRating);

    const avgNetRating = seasonRanking.reduce((s, t) => s + t.netRating, 0) / Math.max(1, seasonRanking.length);

    for (let idx = 0; idx < seasonRanking.length; idx++) {
      const stat = seasonRanking[idx];
      const match = teamsByNorm.get(normalize(stat.teamName));
      if (!match) continue;

      const pace = stat.pace;
      const possessions = Math.max(50, Math.min(90, pace));

      const adjOE = stat.pointsFor / Math.max(1, stat.games) / possessions * 100;
      const adjDE = stat.pointsAgainst / Math.max(1, stat.games) / possessions * 100;
      const netRating = adjOE - adjDE;

      // Approximate seed based on ranking within the season
      const seed = Math.min(16, Math.max(1, Math.ceil((idx + 1) / 4)));

      // Approximate SOS as a slight adjustment around 50 based on net rating relative to average
      const sos = Math.max(1, Math.min(99, 50 + (netRating - avgNetRating) * 0.6));

      // Historic 'last 10' approximated from full-season win rate
      const last10 = Math.round((stat.wins / Math.max(1, stat.games)) * 10);

      out.set(`${match.id}-${season}`, {
        ...match,
        adjOE,
        adjDE,
        adjTempo: pace,
        wins: stat.wins,
        losses: stat.losses,
        netRanking: idx + 1,
        sos,
        last10,
        efgPct: 0.52,
        toRate: 16,
        orbPct: 28,
        ftRate: 32,
        defEfgPct: 0.50,
        defToRate: 16,
        defOrbPct: 28,
        defFtRate: 32,
        threePtRate: 0.35,
        threePtPct: 0.34,
        experience: 0.5,
        coachTourneyWins: 0,
        seed,
        roster: match.roster,
      });
    }
  }

  return out;
}

export async function buildHistoricalTeamSnapshots(
  teams: Team[],
  seasons: number[],
): Promise<Map<string, Team>> {
  const stats = await fetchHistoricalSeasonStats(seasons);
  return mapHistoricalStatsToTeams(teams, stats);
}

export function buildHistoricalSamples(
  teams: Team[],
  outcomes: GameOutcome[],
  sentiment?: Map<string, number>,
  seasonTeams?: Map<string, Team>,
): Array<{ features: number[]; label: number }> {
  const normalize = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
  const teamMap = new Map<string, Team>();
  teams.forEach(t => {
    teamMap.set(normalize(t.name), t);
    if (t.abbreviation) teamMap.set(normalize(t.abbreviation), t);
  });

  const samples: Array<{ features: number[]; label: number }> = [];

  for (const out of outcomes) {
    const a = teamMap.get(normalize(out.teamA));
    const b = teamMap.get(normalize(out.teamB));
    if (!a || !b) continue;

    const seasonA = seasonTeams?.get(`${a.id}-${out.season}`) ?? a;
    const seasonB = seasonTeams?.get(`${b.id}-${out.season}`) ?? b;

    const label = normalize(out.winner) === normalize(out.teamA) ? 1 : 0;
    const features = [
      ...computeFeatures(seasonA, seasonB, sentiment),
    ];

    samples.push({ features, label });
  }

  return samples;
}
