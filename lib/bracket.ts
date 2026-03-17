import { TEAMS, getTeamsByRegion } from './teams';
import type { Team } from './types';

export const REGIONS = ['East', 'South', 'Midwest', 'West'] as const;

/** Returns 4 arrays of 16 teams, each in standard NCAA bracket seeding order.
 *  Index 0=East, 1=South, 2=Midwest, 3=West */
export function buildRegionTeams(): Team[][] {
  return REGIONS.map(r => getTeamsByRegion(r));
}

/** Slot height in pixels — base unit for bracket layout math */
export const SLOT_H = 28;

/** Width of each round column in pixels */
export const ROUND_W = 148;

/** Gap between round columns in pixels */
export const COL_GAP = 2;

/** Total height of a single region bracket (16 teams × SLOT_H per team slot) */
export const REGION_H = 16 * SLOT_H; // 448px

/** Given a round index (0=R64…3=E8) and game index within that round,
 *  compute the absolute top-offset in px for that game's container.
 *  Formula: center of game = midpoint between its two source games in round-1.
 *  y = gameIdx * (REGION_H / numGames) + (REGION_H / numGames - SLOT_H * 2) / 2
 */
export function gameTopPx(roundIdx: number, gameIdx: number): number {
  const numGames = 8 / Math.pow(2, roundIdx); // 8,4,2,1
  const slotHeight = REGION_H / numGames;      // height allocated per game
  const gameHeight = SLOT_H * 2;               // each game shows 2 team slots
  return gameIdx * slotHeight + (slotHeight - gameHeight) / 2;
}

/** Seed color by bucket */
export function seedColor(seed: number): string {
  if (seed === 1) return '#f59e0b';   // gold
  if (seed <= 4) return '#60a5fa';   // blue
  if (seed <= 8) return '#a3e635';   // green
  if (seed <= 12) return '#fb923c';  // orange
  if (seed <= 15) return '#f472b6';  // pink
  return '#94a3b8';                  // gray (16)
}

export function winProbColor(prob: number): string {
  if (prob >= 0.75) return '#22c55e';
  if (prob >= 0.55) return '#84cc16';
  if (prob >= 0.45) return '#f59e0b';
  if (prob >= 0.30) return '#f97316';
  return '#ef4444';
}
