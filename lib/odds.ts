import type { Team } from './types';

// Strict key for matchup between two teams (order matters: a vs b)
export function buildMatchKey(a: Team, b: Team): string {
  return `${a.id}__vs__${b.id}`;
}

function normalizeName(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
}

function findTeamByName(teams: Team[], name: string): Team | undefined {
  const norm = normalizeName(name);
  return teams.find(t => normalizeName(t.name) === norm || normalizeName(t.abbreviation ?? '') === norm);
}

export async function fetchMarketOdds(teams: Team[]): Promise<Map<string, number>> {
  const odds = new Map<string, number>();
  const apiKey = process.env.NEXT_PUBLIC_ODDS_API_KEY;
  if (!apiKey) return odds;

  try {
    const res = await fetch(
      `https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds?regions=us&markets=h2h&oddsFormat=american&dateFormat=unix&apiKey=${apiKey}`
    );
    if (!res.ok) return odds;

    const data = await res.json();
    if (!Array.isArray(data)) return odds;

    for (const game of data) {
      const { teams: gameTeams, odds: gameOdds } = game as any;
      if (!Array.isArray(gameTeams) || !Array.isArray(gameOdds)) continue;

      const h2h = gameOdds.find((o: any) => o.market_key === 'h2h');
      if (!h2h || !Array.isArray(h2h.outcomes)) continue;

      for (const outcome of h2h.outcomes) {
        const teamName: string = outcome.name;
        const american: number = outcome.price;
        const team = findTeamByName(teams, teamName);
        if (!team) continue;

        // Reserve the probability mapping using implied probability
        const prob = americanToProb(american);
        // For each game, store it under both directions for easy lookup
        const otherTeamName = gameTeams.find((n: string) => normalizeName(n) !== normalizeName(teamName));
        if (!otherTeamName) continue;
        const otherTeam = findTeamByName(teams, otherTeamName);
        if (!otherTeam) continue;

        const keyA = buildMatchKey(team, otherTeam);
        const keyB = buildMatchKey(otherTeam, team);
        odds.set(keyA, prob);
        odds.set(keyB, 1 - prob);
      }
    }
  } catch (e) {
    // ignore failures; fall back to seed-based odds
  }

  return odds;
}

function americanToProb(american: number): number {
  if (american > 0) {
    return 100 / (american + 100);
  }
  return -american / (-american + 100);
}
