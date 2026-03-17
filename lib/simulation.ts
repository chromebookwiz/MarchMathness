import type { Team, ModelWeights, ModelStats, FeatureImportance, TrainingProgress } from './types';
import type { DeepNNWeights } from './neuralNet';
import { nnPredictDeep } from './neuralNet';
import { playerWinProbAdjustment } from './playerSim';

// ── Randomness ─────────────────────────────────────────────────────────────
export let globalSeed = 2026;
export function setSeed(s: number) { globalSeed = s; }
export function rand(): number {
  globalSeed = (globalSeed * 1664525 + 1013904223) | 0;
  return (globalSeed >>> 0) / 4294967296;
}

// ─── Feature names ─────────────────────────────────────────────────────────

export const FEATURE_NAMES = [
  'Efficiency Margin',
  'Adj. Offensive Eff.',
  'Adj. Defensive Eff.',
  'Adjusted Tempo',
  'NET Ranking',
  'Q1 Win Rate',
  'Overall Win Rate',
  'Recent Form (L10)',
  'Effective FG%',
  'Turnover Rate',
  'Off. Rebound Rate',
  'Free Throw Rate',
  'Strength of Schedule',
  'Experience Factor',
  'Coach Tournament CV',
  'Defensive Score',
  'Raw Scoring Margin',
  'Q1+Q2 Win Rate',
];

export const NUM_FEATURES = FEATURE_NAMES.length; // 18

// ─── Feature engineering ───────────────────────────────────────────────────

function w(a: number, b: number) { return a / (a + b + 1e-9); }
function q1r(t: Team) { return w(t.q1W, t.q1L); }
function q12r(t: Team) { return w(t.q1W + t.q2W, t.q1L + t.q2L); }
function defScore(t: Team) {
  return (110 - t.adjDE) * 0.4 + t.defToRate * 0.35 + (100 - t.defEfgPct) * 0.5;
}

export function computeFeatures(a: Team, b: Team): number[] {
  const aEM = a.adjOE - a.adjDE;
  const bEM = b.adjOE - b.adjDE;
  return [
    (aEM - bEM) * 0.08,
    (a.adjOE - b.adjOE) * 0.04,
    (b.adjDE - a.adjDE) * 0.04,
    (a.adjTempo - b.adjTempo) * 0.015,
    (b.netRanking - a.netRanking) * 0.004,
    q1r(a) - q1r(b),
    w(a.wins, a.losses) - w(b.wins, b.losses),
    (a.last10 - b.last10) / 10,
    (a.efgPct - b.efgPct) * 0.012,
    (b.toRate - a.toRate) * 0.06,
    (a.orbPct - b.orbPct) * 0.012,
    (a.ftRate - b.ftRate) * 0.008,
    (b.sos - a.sos) * 0.003,
    (a.experience - b.experience) * 0.4,
    (a.coachTourneyWins - b.coachTourneyWins) * 0.008,
    (defScore(a) - defScore(b)) * 0.03,
    ((aEM - bEM) / 30),
    q12r(a) - q12r(b),
  ];
}

function sigmoid(x: number) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

// ─── Logistic regression predict ──────────────────────────────────────────

export function lrPredict(weights: ModelWeights, a: Team, b: Team): number {
  const f = computeFeatures(a, b);
  return sigmoid(f.reduce((s, fi, i) => s + weights[i] * fi, 0));
}

// ─── Elo rating system ────────────────────────────────────────────────────

export function buildEloRatings(teams: Team[]): Map<string, number> {
  const elos = new Map<string, number>();
  for (const t of teams) {
    // Base Elo from NET ranking + efficiency margin
    const em = t.adjOE - t.adjDE;
    const netBonus = Math.max(0, (80 - t.netRanking)) * 6;
    const emBonus  = em * 13;
    const sosBonus = Math.max(0, (90 - t.sos)) * 1.5;
    elos.set(t.id, 1500 + netBonus + emBonus + sosBonus);
  }
  return elos;
}

export function eloWinProb(eloa: number, elob: number): number {
  return 1 / (1 + Math.pow(10, (elob - eloa) / 400));
}

// ─── Ensemble win probability ─────────────────────────────────────────────

export interface EnsembleModel {
  lrWeights: ModelWeights;
  nnWeights: DeepNNWeights;
  elos: Map<string, number>;
  marketOdds?: Map<string, number>;
  ensembleW: { lr: number; nn: number; elo: number; em: number; market: number };
}

function marketWinProb(a: Team, b: Team, marketOdds?: Map<string, number>): number {
  if (marketOdds) {
    const key = `${a.id}__vs__${b.id}`;
    const p = marketOdds.get(key);
    if (p != null) return p;
  }

  // Fallback proxy: betting markets typically favor lower seeds.
  const seedDiff = b.seed - a.seed;
  return 1 / (1 + Math.exp(-seedDiff / 4));
}

export function ensembleWinProb(
  model: EnsembleModel,
  a: Team,
  b: Team,
): number {
  const { lr, nn, elo, em, market } = model.ensembleW;

  const lrP = lrPredict(model.lrWeights, a, b);
  const nnP = nnPredictDeep(model.nnWeights, computeFeatures(a, b));

  const eloA = model.elos.get(a.id) ?? 1500;
  const eloB = model.elos.get(b.id) ?? 1500;
  const eloP = eloWinProb(eloA, eloB);

  const aEM = a.adjOE - a.adjDE;
  const bEM = b.adjOE - b.adjDE;
  const emP = sigmoid((aEM - bEM) * 0.20);

  const marketP = marketWinProb(a, b, model.marketOdds);

  // Base ensemble (weighted average)
  const totalW = lr + nn + elo + em + market;
  let p = (lr * lrP + nn * nnP + elo * eloP + em * emP + market * marketP) / Math.max(totalW, 1);

  // Player-level adjustment (if roster data available)
  const playerAdj = playerWinProbAdjustment(a, b);
  p = Math.max(0.03, Math.min(0.97, p + playerAdj));

  return p;
}

// ─── Training data generation ──────────────────────────────────────────────

export interface GameSample { features: number[]; label: number; }

export function generateTrainingSamples(teams: Team[]): GameSample[] {
  const samples: GameSample[] = [];
  const priorW = [2.0, 1.1, 1.1, 0.08, 0.6, 1.0, 0.7, 0.8, 0.5, 0.6, 0.25, 0.12, 0.35, 0.3, 0.18, 0.65, 1.0, 0.85];

  function synOpp(q: 1 | 2 | 3 | 4): Team {
    const baseOE = [122, 116, 110, 104][q - 1] + (rand() - 0.5) * 6;
    const baseDE = [93, 101, 107, 114][q - 1] + (rand() - 0.5) * 6;
    return {
      id: 'opp', name: 'Opp', seed: 0, region: 'East', espnId: 0,
      adjOE: baseOE, adjDE: baseDE, adjTempo: 67 + (rand() - 0.5) * 8,
      wins: 18 + Math.floor(rand() * 8), losses: 5 + Math.floor(rand() * 10),
      q1W: [6,3,1,0][q-1], q1L: [6,4,2,1][q-1],
      q2W: 5, q2L: 3, q3W: 6, q3L: 1, q4W: 4, q4L: 0,
      netRanking: [20,55,115,200][q-1] + (rand() - 0.5) * 25,
      sos: [25,55,110,185][q-1],
      last10: 6 + Math.floor(rand() * 4),
      efgPct: 51 + (rand() - 0.5) * 5,
      toRate: 16 + (rand() - 0.5) * 4,
      orbPct: 28 + (rand() - 0.5) * 6,
      ftRate: 35 + (rand() - 0.5) * 6,
      defEfgPct: 52, defToRate: 17, defOrbPct: 28, defFtRate: 33,
      threePtRate: 38, threePtPct: 35,
      experience: 0.4 + rand() * 0.3,
      coachTourneyWins: Math.floor(rand() * 20),
    };
  }

  teams.forEach(team => {
    const nr = team.netRanking;
    const dist: [number, number, number, number] =
      nr <= 15 ? [14,8,6,4] : nr <= 35 ? [11,9,7,5] : nr <= 65 ? [8,9,9,6] : [4,7,11,10];

    for (let q = 0; q < 4; q++) {
      const band = (q + 1) as 1 | 2 | 3 | 4;
      for (let g = 0; g < dist[q]; g++) {
        const opp = synOpp(band);
        const features = computeFeatures(team, opp);
        const z = features.reduce((s, f, i) => s + priorW[i] * f, 0);
        const prob = sigmoid(z);
        const noisy = Math.max(0.04, Math.min(0.96, prob + (rand() - 0.5) * 0.22));
        samples.push({ features, label: rand() < noisy ? 1 : 0 });
      }
    }

    // Direct H2H matchups vs nearby-ranked teams
    const peers = teams.filter(t => t.id !== team.id && Math.abs(t.netRanking - team.netRanking) < 22);
    const numH2H = Math.min(6, peers.length);
    for (let i = 0; i < numH2H; i++) {
      const opp = peers[Math.floor(rand() * peers.length)];
      const features = computeFeatures(team, opp);
      const z = features.reduce((s, f, i) => s + priorW[i] * f, 0);
      const prob = sigmoid(z);
      const noisy = Math.max(0.05, Math.min(0.95, prob + (rand() - 0.5) * 0.18));
      samples.push({ features, label: rand() < noisy ? 1 : 0 });
    }
  });

  // Shuffle
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }
  return samples;
}

// ─── Logistic regression (Adam) ────────────────────────────────────────────

export async function trainLogisticRegression(
  samples: GameSample[],
  onProgress: (p: TrainingProgress) => void,
): Promise<ModelWeights> {
  const EPOCHS = 2000;
  const LR = 0.015;
  const B1 = 0.9, B2 = 0.999, EPS = 1e-8, LAMBDA = 0.0008;
  const BATCH = 64;
  const n = samples.length;

  const weights = Array.from({ length: NUM_FEATURES }, () => (rand() - 0.5) * 0.05);
  const m = new Array(NUM_FEATURES).fill(0);
  const v = new Array(NUM_FEATURES).fill(0);
  let t = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    let totalLoss = 0; let correct = 0;
    const lr = LR * (0.5 + 0.5 * Math.cos((epoch / EPOCHS) * Math.PI));

    for (let b = 0; b < n; b += BATCH) {
      const bEnd = Math.min(b + BATCH, n);
      const grad = new Array(NUM_FEATURES).fill(0);

      for (let si = b; si < bEnd; si++) {
        const { features, label } = samples[si];
        const z = features.reduce((s, f, i) => s + weights[i] * f, 0);
        const p = sigmoid(z);
        totalLoss += -(label * Math.log(p + 1e-10) + (1 - label) * Math.log(1 - p + 1e-10));
        if ((p > 0.5) === (label === 1)) correct++;
        const err = p - label;
        features.forEach((f, i) => { grad[i] += err * f; });
      }

      const bSize = bEnd - b;
      t++;
      for (let i = 0; i < NUM_FEATURES; i++) {
        const g = grad[i] / bSize + LAMBDA * weights[i];
        m[i] = B1 * m[i] + (1 - B1) * g;
        v[i] = B2 * v[i] + (1 - B2) * g * g;
        const mH = m[i] / (1 - Math.pow(B1, t));
        const vH = v[i] / (1 - Math.pow(B2, t));
        weights[i] -= lr * mH / (Math.sqrt(vH) + EPS);
      }
    }

    if (epoch % 40 === 0 || epoch === EPOCHS - 1) {
      onProgress({
        epoch: epoch + 1,
        totalEpochs: EPOCHS,
        loss: totalLoss / n,
        accuracy: correct / n,
        lrDecay: 0.5 + 0.5 * Math.cos((epoch / EPOCHS) * Math.PI),
        modelType: 'logistic',
      });
      await new Promise(r => setTimeout(r, 0));
    }
  }
  return weights;
}

// ─── Model stats ───────────────────────────────────────────────────────────

export function computeModelStats(
  weights: ModelWeights,
  samples: GameSample[],
  finalLoss: number,
  finalAccuracy: number,
  nnAcc: number,
): ModelStats {
  const fi: FeatureImportance[] = weights
    .map((w, i) => ({ name: FEATURE_NAMES[i], weight: w, absWeight: Math.abs(w), rank: 0 }))
    .sort((a, b) => b.absWeight - a.absWeight)
    .map((f, idx) => ({ ...f, rank: idx + 1 }));

  return {
    weights,
    finalLoss,
    finalAccuracy,
    nnAccuracy: nnAcc,
    featureImportance: fi,
    trainingSamples: samples.length,
    epochs: 2000,
    ensembleWeights: { lr: 0.35, nn: 0.35, elo: 0.20, em: 0.10, market: 0.10 },
  };
}

// ─── Bracket simulation ────────────────────────────────────────────────────

export interface BracketSimOutput {
  regionRounds: Team[][][];
  finalFour: Team[];
  ffWinners: Team[];
  champion: Team;
  allRoundWinners: Map<string, Team>;
  winProbs: Map<string, number>;
}

export function simulateBracket(
  regionTeams: Team[][],
  model: EnsembleModel,
  abortSignal?: { aborted: boolean },
): BracketSimOutput {
  const regionRounds: Team[][][] = [[], [], [], []];
  const finalFour: Team[] = [];
  const allRoundWinners = new Map<string, Team>();
  const winProbs = new Map<string, number>();

  for (let r = 0; r < 4; r++) {
    let pool = [...regionTeams[r]];
    regionRounds[r] = [];

    for (let round = 0; round < 4; round++) {
      const next: Team[] = [];
      for (let i = 0; i < pool.length; i += 2) {
        if (abortSignal?.aborted) return { regionRounds, finalFour: [], ffWinners: [], champion: pool[0], allRoundWinners, winProbs };
        const a = pool[i], b = pool[i + 1];
        const prob = ensembleWinProb(model, a, b);
        const winner = prob >= 0.5 ? a : b;
        const key = `r${r}_rd${round}_g${i / 2}`;
        allRoundWinners.set(key, winner);
        winProbs.set(key, winner === a ? prob : 1 - prob);
        next.push(winner);
      }
      regionRounds[r].push(next);
      pool = next;
    }
    finalFour.push(pool[0]);
  }

  // Final Four: East vs South, Midwest vs West
  const ff0Prob = ensembleWinProb(model, finalFour[0], finalFour[1]);
  const ff1Prob = ensembleWinProb(model, finalFour[2], finalFour[3]);
  const ff0 = ff0Prob >= 0.5 ? finalFour[0] : finalFour[1];
  const ff1 = ff1Prob >= 0.5 ? finalFour[2] : finalFour[3];

  const champProb = ensembleWinProb(model, ff0, ff1);
  const champ = champProb >= 0.5 ? ff0 : ff1;

  return {
    regionRounds,
    finalFour,
    ffWinners: [ff0, ff1],
    champion: champ,
    allRoundWinners,
    winProbs,
  };
}


// ─── Display bracket builder ───────────────────────────────────────────────

export interface DisplayGame {
  id: string;
  teamA: Team | null;
  teamB: Team | null;
  winner: Team | null;
  winProbA: number;
  marketProbA?: number;
  marketProbB?: number;
  round: number;
  region: string;
  position: number;
  teamAConsensus?: number;
  teamBConsensus?: number;
}

export interface DisplayBracket {
  east: DisplayGame[][];
  south: DisplayGame[][];
  midwest: DisplayGame[][];
  west: DisplayGame[][];
  finalFour: DisplayGame[];
  championship: DisplayGame;
}

function buildRegionGames(
  regionTeams: Team[],
  regionName: string,
  model: EnsembleModel,
  simResult: BracketSimOutput,
  regionIdx: number,
): DisplayGame[][] {
  const rounds: DisplayGame[][] = [];
  let pool = [...regionTeams];
  const pools: Team[][] = [pool];

  for (let rd = 0; rd < 3; rd++) {
    const next: Team[] = [];
    for (let i = 0; i < pool.length; i += 2) {
      const key = `r${regionIdx}_rd${rd}_g${i / 2}`;
      next.push(simResult.allRoundWinners.get(key) ?? pool[i]);
    }
    pool = next;
    pools.push([...pool]);
  }

  for (let rd = 0; rd < 4; rd++) {
    const p0 = pools[rd];
    const games: DisplayGame[] = [];
    for (let i = 0; i < p0.length; i += 2) {
      const a = p0[i], b = p0[i + 1];
      const key = `r${regionIdx}_rd${rd}_g${i / 2}`;
      const winner = simResult.allRoundWinners.get(key) ?? null;
      const prob = ensembleWinProb(model, a, b);
      const marketP = marketWinProb(a, b);
      games.push({
        id: key, teamA: a, teamB: b, winner, winProbA: prob,
        marketProbA: marketP, marketProbB: 1 - marketP,
        round: rd + 1, region: regionName, position: i / 2,
      });
    }
    rounds.push(games);
  }
  return rounds;
}

export function buildDisplayBracket(
  regionTeams: Team[][],
  model: EnsembleModel,
  simResult: BracketSimOutput,
): DisplayBracket {
  const names = ['East', 'South', 'Midwest', 'West'];
  const [east, south, midwest, west] = names.map((name, idx) =>
    buildRegionGames(regionTeams[idx], name, model, simResult, idx)
  );

  const [ff0a, ff0b, ff1a, ff1b] = [
    simResult.finalFour[0], simResult.finalFour[1],
    simResult.finalFour[2], simResult.finalFour[3],
  ];



  const champA = simResult.ffWinners[0];
  const champB = simResult.ffWinners[1];

  return {
    east, south, midwest, west,
    finalFour: [
      {
        id: 'ff_0', teamA: ff0a, teamB: ff0b, winner: simResult.ffWinners[0],
        winProbA: ensembleWinProb(model, ff0a, ff0b),
        marketProbA: marketWinProb(ff0a, ff0b),
        marketProbB: 1 - marketWinProb(ff0a, ff0b),
        round: 5, region: 'Final Four', position: 0,
      },
      {
        id: 'ff_1', teamA: ff1a, teamB: ff1b, winner: simResult.ffWinners[1],
        winProbA: ensembleWinProb(model, ff1a, ff1b),
        marketProbA: marketWinProb(ff1a, ff1b),
        marketProbB: 1 - marketWinProb(ff1a, ff1b),
        round: 5, region: 'Final Four', position: 1,
      },
    ],
    championship: {
      id: 'champ', teamA: champA, teamB: champB, winner: simResult.champion,
      winProbA: ensembleWinProb(model, champA, champB),
      marketProbA: marketWinProb(champA, champB),
      marketProbB: 1 - marketWinProb(champA, champB),
      round: 6, region: 'Championship', position: 0,
    },
  };
}
