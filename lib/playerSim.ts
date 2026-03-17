import type { Team, Player } from './types';

/**
 * Player-level possession simulation.
 * Used for enriching team-level features and for detailed game analysis.
 */

export interface PossessionResult {
  points: number;
  scorer: Player | null;
  assister: Player | null;
  turnoverPlayer: Player | null;
  foulPlayer: Player | null;
  rebounder: Player | null;
  shotType: '2PT' | '3PT' | 'FT' | 'TOV' | null;
  made: boolean;
}

export interface PlayerGameLine {
  player: Player;
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  tov: number;
  fls: number;
  fga: number;
  fgm: number;
  fg3a: number;
  fg3m: number;
  fta: number;
  ftm: number;
  min: number;
}

export interface FullGameResult {
  teamAScore: number;
  teamBScore: number;
  teamALines: PlayerGameLine[];
  teamBLines: PlayerGameLine[];
  totalPossessions: number;
  overtime: boolean;
  mvp: Player | null;
}

function pickPlayer(players: Player[]): Player {
  // Weight by usage rate
  const total = players.reduce((s, p) => s + p.usageRate, 0) + 1e-9;
  let r = Math.random() * total;
  for (const p of players) {
    r -= p.usageRate;
    if (r <= 0) return p;
  }
  return players[players.length - 1];
}

function gaussNoise(mean: number, sd: number): number {
  // Box-Muller
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shootingProb(
  shooter: Player,
  shotType: '2PT' | '3PT',
  defTeam: Team,
): number {
  const leagueAvgEFG = 0.515;
  const defMult = leagueAvgEFG / (defTeam.defEfgPct / 100 + 0.001);

  if (shotType === '3PT') {
    return Math.max(0.15, Math.min(0.65,
      shooter.fg3Pct * defMult * gaussNoise(1.0, 0.08)
    ));
  }
  // 2PT: compute from overall FG% minus 3PT contribution
  const tw2Pct = shooter.fgaPerGame > 0
    ? (shooter.fgPct * shooter.fgaPerGame - shooter.fg3Pct * shooter.fg3aPerGame)
      / (shooter.fgaPerGame - shooter.fg3aPerGame + 0.001)
    : 0.48;
  return Math.max(0.25, Math.min(0.75,
    tw2Pct * defMult * gaussNoise(1.0, 0.07)
  ));
}

function simulatePossession(
  offTeamPlayers: Player[],
  defTeam: Team,
  defPlayers: Player[],
  lines: Map<string, PlayerGameLine>,
): number {
  // Turnover check
  const toProb = (offTeamPlayers.reduce((s, p) => s + p.topg, 0) / (offTeamPlayers.length || 1)) / 100;
  if (Math.random() < toProb) {
    const turnoverPlayer = pickPlayer(offTeamPlayers);
    const line = lines.get(turnoverPlayer.id);
    if (line) line.tov++;
    // Steal opportunity for defense
    const stealer = defPlayers[Math.floor(Math.random() * defPlayers.length)];
    const defLine = lines.get(stealer?.id ?? '');
    if (defLine && Math.random() < 0.4) defLine.stl++;
    return 0;
  }

  const shooter = pickPlayer(offTeamPlayers);
  const sLine = lines.get(shooter.id);
  if (!sLine) return 0;

  // Decide 3PT vs 2PT
  const three = shooter.fgaPerGame > 0
    ? shooter.fg3aPerGame / shooter.fgaPerGame
    : 0.35;
  const shotType: '2PT' | '3PT' = Math.random() < three ? '3PT' : '2PT';

  if (sLine) {
    shotType === '3PT' ? sLine.fg3a++ : sLine.fga++;
  }

  // Foul check
  const foulProb = shotType === '3PT' ? 0.025 : 0.06;
  if (Math.random() < foulProb) {
    // Free throws
    const numFTs = shotType === '3PT' ? 3 : 2;
    let pts = 0;
    if (sLine) sLine.fta += numFTs;
    for (let i = 0; i < numFTs; i++) {
      if (Math.random() < shooter.ftPct) {
        pts++;
        if (sLine) sLine.ftm++;
      }
    }
    // Possible and-1
    if (shotType === '2PT' && Math.random() < 0.2) {
      const madeShot = Math.random() < shootingProb(shooter, '2PT', defTeam);
      if (madeShot) {
        pts += 2;
        if (sLine) { sLine.fgm++; }
      }
    }
    if (sLine) sLine.pts += pts;
    return pts;
  }

  // Blocked shot
  const topBlocker = defPlayers.reduce((best, p) => p.bpg > best.bpg ? p : best, defPlayers[0]);
  const blockProb = (topBlocker?.bpg ?? 0) / 80;
  if (topBlocker && Math.random() < blockProb) {
    const bLine = lines.get(topBlocker.id);
    if (bLine) bLine.blk++;
    // Offensive rebound after block
    if (Math.random() < 0.25) {
      const rebounder = offTeamPlayers[Math.floor(Math.random() * offTeamPlayers.length)];
      const rLine = lines.get(rebounder.id);
      if (rLine) rLine.reb++;
      // Kick-out 3
      if (Math.random() < 0.4) {
        const shooter2 = pickPlayer(offTeamPlayers);
        const made = Math.random() < shooter2.fg3Pct * 0.9;
        const s2Line = lines.get(shooter2.id);
        if (s2Line) { s2Line.fg3a++; if (made) { s2Line.fg3m++; s2Line.fgm++; s2Line.pts += 3; } }
        return made ? 3 : 0;
      }
    }
    return 0;
  }

  // Shot outcome
  const made = Math.random() < shootingProb(shooter, shotType, defTeam);

  if (made) {
    const pts = shotType === '3PT' ? 3 : 2;
    if (sLine) {
      sLine.fgm++;
      if (shotType === '3PT') sLine.fg3m++;
      sLine.pts += pts;
    }
    // Assist opportunity
    if (Math.random() < 0.55) {
      const potAssists = offTeamPlayers.filter(p => p.id !== shooter.id);
      if (potAssists.length > 0) {
        const assister = potAssists.reduce((best, p) => p.apg > best.apg ? p : best, potAssists[0]);
        const aLine = lines.get(assister.id);
        if (aLine) aLine.ast++;
      }
    }
    return pts;
  } else {
    // Missed shot → rebound
    const orbPct = defTeam.defOrbPct / 100;
    if (Math.random() < (1 - orbPct)) {
      // Defensive rebound
      const rebounder = defPlayers[Math.floor(Math.random() * defPlayers.length)];
      const rLine = lines.get(rebounder?.id ?? '');
      if (rLine) rLine.reb++;
    } else {
      // Offensive rebound
      const rebounder = offTeamPlayers[Math.floor(Math.random() * offTeamPlayers.length)];
      const rLine = lines.get(rebounder.id);
      if (rLine) rLine.reb++;
      // Put-back 2
      const putbackMade = Math.random() < 0.52;
      if (putbackMade) {
        const rLine2 = lines.get(rebounder.id);
        if (rLine2) { rLine2.fga++; rLine2.fgm++; rLine2.pts += 2; }
        return 2;
      }
    }
    return 0;
  }
}

export function simulateFullGame(teamA: Team, teamB: Team): FullGameResult {
  const playersA = (teamA.roster ?? []).slice(0, 10);
  const playersB = (teamB.roster ?? []).slice(0, 10);

  if (!playersA.length || !playersB.length) {
    // Fallback: use team-level stats to estimate score
    const aEM = (teamA.adjOE - teamA.adjDE) / 2;
    const bEM = (teamB.adjOE - teamB.adjDE) / 2;
    const aScore = Math.round(gaussNoise(68 + aEM, 6));
    const bScore = Math.round(gaussNoise(68 + bEM, 6));
    return {
      teamAScore: Math.max(40, aScore),
      teamBScore: Math.max(40, bScore),
      teamALines: [],
      teamBLines: [],
      totalPossessions: 65,
      overtime: false,
      mvp: null,
    };
  }

  // Init box score lines
  const linesA = new Map<string, PlayerGameLine>();
  const linesB = new Map<string, PlayerGameLine>();

  function initLine(p: Player): PlayerGameLine {
    return { player: p, pts:0, reb:0, ast:0, stl:0, blk:0, tov:0, fls:0,
             fga:0, fgm:0, fg3a:0, fg3m:0, fta:0, ftm:0, min:0 };
  }
  playersA.forEach(p => linesA.set(p.id, initLine(p)));
  playersB.forEach(p => linesB.set(p.id, initLine(p)));

  // Number of possessions per team (based on tempo average)
  const avgTempo = (teamA.adjTempo + teamB.adjTempo) / 2;
  const possPerTeam = Math.round(gaussNoise(avgTempo * 0.94, 3));

  let scoreA = 0;
  let scoreB = 0;

  // Active players (simplified: all 10 play, weighted by mpg for minutes)
  const activeA = playersA.slice(0, 8);
  const activeB = playersB.slice(0, 8);

  for (let i = 0; i < possPerTeam; i++) {
    scoreA += simulatePossession(activeA, teamB, activeB, linesA);
    scoreB += simulatePossession(activeB, teamA, activeA, linesB);
  }

  // Overtime if tied
  let overtime = false;
  if (scoreA === scoreB) {
    overtime = true;
    for (let i = 0; i < 5; i++) {
      scoreA += simulatePossession(activeA, teamB, activeB, linesA);
      scoreB += simulatePossession(activeB, teamA, activeA, linesB);
    }
  }

  // Add minutes (approx)
  playersA.forEach((p, i) => {
    const line = linesA.get(p.id);
    if (line) line.min = Math.round(p.mpg * gaussNoise(1.0, 0.1));
  });
  playersB.forEach((p, i) => {
    const line = linesB.get(p.id);
    if (line) line.min = Math.round(p.mpg * gaussNoise(1.0, 0.1));
  });

  const allLines = [
    ...Array.from(linesA.values()),
    ...Array.from(linesB.values()),
  ];

  const mvp = allLines.reduce((best, l) =>
    (l.pts + l.reb * 1.2 + l.ast * 1.5) > (best.pts + best.reb * 1.2 + best.ast * 1.5)
      ? l : best
  , allLines[0])?.player ?? null;

  return {
    teamAScore: scoreA,
    teamBScore: scoreB,
    teamALines: Array.from(linesA.values()).sort((a, b) => b.pts - a.pts),
    teamBLines: Array.from(linesB.values()).sort((a, b) => b.pts - a.pts),
    totalPossessions: possPerTeam * 2,
    overtime,
    mvp,
  };
}

// ── Compute player-enhanced win probability adjustment ────────────────────

function parseHeightToInches(ht: string): number {
  const m = ht.match(/(\d+)'(\d+)/);
  if (!m) return 75; // 6'3"
  return parseInt(m[1], 10) * 12 + parseInt(m[2], 10);
}

function estimateReach(heightInches: number, position: string): number {
  // Wingspan/reach estimation based on height and position archetype
  let wingspanBonus = 2; // Guards +2 inches
  if (position === 'F' || position === 'SF') wingspanBonus = 4;
  if (position === 'PF') wingspanBonus = 5;
  if (position === 'C') wingspanBonus = 6;
  
  // Standing reach roughly height + wingspanBonus * 0.5 + reach constant
  return heightInches * 1.33 + wingspanBonus;
}

export function playerWinProbAdjustment(teamA: Team, teamB: Team): number {
  const rA = teamA.roster ?? [];
  const rB = teamB.roster ?? [];
  if (!rA.length || !rB.length) return 0;

  // Weight players by minutes played
  const totalMinA = rA.reduce((s, p) => s + p.mpg, 0) || 1;
  const totalMinB = rB.reduce((s, p) => s + p.mpg, 0) || 1;

  // 1. Minutes-weighted height & reach
  let hgtA = 0, reachA = 0;
  rA.forEach(p => { 
    const h = parseHeightToInches(p.height);
    const r = estimateReach(h, p.position);
    hgtA += h * (p.mpg / totalMinA);
    reachA += r * (p.mpg / totalMinA);
  });

  let hgtB = 0, reachB = 0;
  rB.forEach(p => { 
    const h = parseHeightToInches(p.height);
    const r = estimateReach(h, p.position);
    hgtB += h * (p.mpg / totalMinB);
    reachB += r * (p.mpg / totalMinB);
  });

  const heightAdv = (hgtA - hgtB) / 10; // 1 inch = 0.1% win prob
  const reachAdv  = (reachA - reachB) / 10; 

  // 2. Center Matchup (Top Minute Center/PF)
  const bigA = rA.filter(p => p.position === 'C' || p.position === 'PF').sort((a,b) => b.mpg - a.mpg)[0];
  const bigB = rB.filter(p => p.position === 'C' || p.position === 'PF').sort((a,b) => b.mpg - a.mpg)[0];
  
  let bigAdv = 0;
  if (bigA && bigB) {
    const bigHgtA = parseHeightToInches(bigA.height);
    const bigHgtB = parseHeightToInches(bigB.height);
    bigAdv = ((bigHgtA - bigHgtB) + (bigA.bpm - bigB.bpm) * 0.5) * 0.005;
  }

  // 3. Guard Matchup (Ball Security & Playmaking vs Pressure)
  const guardsA = rA.filter(p => p.position.includes('G') || p.position === 'PG').sort((a,b) => b.mpg - a.mpg).slice(0, 2);
  const guardsB = rB.filter(p => p.position.includes('G') || p.position === 'PG').sort((a,b) => b.mpg - a.mpg).slice(0, 2);
  
  const astTovA = guardsA.reduce((s, p) => s + p.apg, 0) / Math.max(0.1, guardsA.reduce((s, p) => s + p.topg, 0));
  const astTovB = guardsB.reduce((s, p) => s + p.apg, 0) / Math.max(0.1, guardsB.reduce((s, p) => s + p.topg, 0));
  const guardAdv = (astTovA - astTovB) * 0.01;

  // 4. Star Power (Top player BPM weighted by usage and minutes)
  const sortA = [...rA].sort((a, b) => (b.bpm * b.mpg * b.usageRate) - (a.bpm * a.mpg * a.usageRate));
  const sortB = [...rB].sort((a, b) => (b.bpm * b.mpg * b.usageRate) - (a.bpm * a.mpg * a.usageRate));
  const starDiff = (sortA[0]?.bpm ?? 0) - (sortB[0]?.bpm ?? 0);

  // 5. Minutes-weighted Depth and Shooting (TS%)
  const wtBpmA = rA.slice(0, 8).reduce((s, p) => s + p.bpm * (p.mpg / totalMinA), 0) * 5;
  const wtBpmB = rB.slice(0, 8).reduce((s, p) => s + p.bpm * (p.mpg / totalMinB), 0) * 5;
  
  const wtTsA = rA.reduce((s, p) => s + p.trueShootingPct * (p.mpg / totalMinA), 0);
  const wtTsB = rB.reduce((s, p) => s + p.trueShootingPct * (p.mpg / totalMinB), 0);
  
  const depthDiff = wtBpmA - wtBpmB;
  const tsDiff = wtTsA - wtTsB;

  // Comprehensive formula integrating height, reach, position matchups, minutes, and stats entirely deterministically
  const adj = (
    heightAdv * 0.01 + 
    reachAdv * 0.015 + 
    bigAdv + 
    guardAdv + 
    starDiff * 0.012 + 
    depthDiff * 0.008 + 
    tsDiff * 0.4
  );

  // Cap adjustment at ±0.08 (8 percentage points)
  return Math.max(-0.08, Math.min(0.08, adj));
}
