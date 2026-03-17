import type { Team, ModelWeights, ModelStats, FeatureImportance, TrainingProgress } from './types';
import type { DeepNNWeights } from './neuralNet';
import { nnPredictDeep } from './neuralNet';
import { playerWinProbAdjustment } from './playerSim';

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
  ensembleW: { lr: number; nn: number; elo: number; em: number };
}

export function ensembleWinProb(
  model: EnsembleModel,
  a: Team,
  b: Team,
): number {
  const { lr, nn, elo, em } = model.ensembleW;

  const lrP = lrPredict(model.lrWeights, a, b);
  const nnP = nnPredictDeep(model.nnWeights, computeFeatures(a, b));

  const eloA = model.elos.get(a.id) ?? 1500;
  const eloB = model.elos.get(b.id) ?? 1500;
  const eloP = eloWinProb(eloA, eloB);

  const aEM = a.adjOE - a.adjDE;
  const bEM = b.adjOE - b.adjDE;
  const emP = sigmoid((aEM - bEM) * 0.20);

  // Base ensemble
  let p = lr * lrP + nn * nnP + elo * eloP + em * emP;

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
    const baseOE = [122, 116, 110, 104][q - 1] + (Math.random() - 0.5) * 6;
    const baseDE = [93, 101, 107, 114][q - 1] + (Math.random() - 0.5) * 6;
    return {
      id: 'opp', name: 'Opp', seed: 0, region: 'East', espnId: 0,
      adjOE: baseOE, adjDE: baseDE, adjTempo: 67 + (Math.random() - 0.5) * 8,
      wins: 18 + Math.floor(Math.random() * 8), losses: 5 + Math.floor(Math.random() * 10),
      q1W: [6,3,1,0][q-1], q1L: [6,4,2,1][q-1],
      q2W: 5, q2L: 3, q3W: 6, q3L: 1, q4W: 4, q4L: 0,
      netRanking: [20,55,115,200][q-1] + (Math.random() - 0.5) * 25,
      sos: [25,55,110,185][q-1],
      last10: 6 + Math.floor(Math.random() * 4),
      efgPct: 51 + (Math.random() - 0.5) * 5,
      toRate: 16 + (Math.random() - 0.5) * 4,
      orbPct: 28 + (Math.random() - 0.5) * 6,
      ftRate: 35 + (Math.random() - 0.5) * 6,
      defEfgPct: 52, defToRate: 17, defOrbPct: 28, defFtRate: 33,
      threePtRate: 38, threePtPct: 35,
      experience: 0.4 + Math.random() * 0.3,
      coachTourneyWins: Math.floor(Math.random() * 20),
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
        const noisy = Math.max(0.04, Math.min(0.96, prob + (Math.random() - 0.5) * 0.22));
        samples.push({ features, label: Math.random() < noisy ? 1 : 0 });
      }
    }

    // Direct H2H matchups vs nearby-ranked teams
    const peers = teams.filter(t => t.id !== team.id && Math.abs(t.netRanking - team.netRanking) < 22);
    const numH2H = Math.min(6, peers.length);
    for (let i = 0; i < numH2H; i++) {
      const opp = peers[Math.floor(Math.random() * peers.length)];
      const features = computeFeatures(team, opp);
      const z = features.reduce((s, f, i) => s + priorW[i] * f, 0);
      const prob = sigmoid(z);
      const noisy = Math.max(0.05, Math.min(0.95, prob + (Math.random() - 0.5) * 0.18));
      samples.push({ features, label: Math.random() < noisy ? 1 : 0 });
    }
  });

  // Shuffle
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
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

  const weights = Array.from({ length: NUM_FEATURES }, () => (Math.random() - 0.5) * 0.05);
  const m = new Array(NUM_FEATURES).fill(0);
  const v = new Array(NUM_FEATURES).fill(0);
  let t = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
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
    ensembleWeights: { lr: 0.35, nn: 0.35, elo: 0.20, em: 0.10 },
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
        const winner = Math.random() < prob ? a : b;
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
  const ff0 = Math.random() < ff0Prob ? finalFour[0] : finalFour[1];
  const ff1 = Math.random() < ff1Prob ? finalFour[2] : finalFour[3];

  const champProb = ensembleWinProb(model, ff0, ff1);
  const champ = Math.random() < champProb ? ff0 : ff1;

  return {
    regionRounds,
    finalFour,
    ffWinners: [ff0, ff1],
    champion: champ,
    allRoundWinners,
    winProbs,
  };
}

// ─── Consensus analysis ────────────────────────────────────────────────────

export interface ConsensusData {
  championFreq: Map<string, number>;
  ffFreq: Map<string, number>;
  e8Freq: Map<string, number>;
  s16Freq: Map<string, number>;
  gameSlotFreq: Map<string, Map<string, number>>;
  totalSims: number;
  mostLikelyBracket: BracketSimOutput;
}

export async function runConsensusAnalysis(
  regionTeams: Team[][],
  model: EnsembleModel,
  numSims: number,
  onProgress: (done: number, total: number) => void,
  abortSignal?: { aborted: boolean },
): Promise<ConsensusData> {
  const championFreq = new Map<string, number>();
  const ffFreq       = new Map<string, number>();
  const e8Freq       = new Map<string, number>();
  const s16Freq      = new Map<string, number>();
  const gameSlotFreq = new Map<string, Map<string, number>>();

  // Track per-slot winner counts for most-likely bracket
  const slotWinnerCounts = new Map<string, Map<string, number>>();

  const BATCH = 150;
  let done = 0;

  while (done < numSims) {
    if (abortSignal?.aborted) break;
    const batch = Math.min(BATCH, numSims - done);
    for (let i = 0; i < batch; i++) {
      const r = simulateBracket(regionTeams, model);

      // Champion
      const cid = r.champion.id;
      championFreq.set(cid, (championFreq.get(cid) ?? 0) + 1);

      // Final Four
      r.finalFour.forEach(t => ffFreq.set(t.id, (ffFreq.get(t.id) ?? 0) + 1));

      // Elite Eight (round index 2 = Sweet 16 winners = E8 participants)
      r.regionRounds.forEach(reg => {
        if (reg[2]) reg[2].forEach(t => e8Freq.set(t.id, (e8Freq.get(t.id) ?? 0) + 1));
        if (reg[1]) reg[1].forEach(t => s16Freq.set(t.id, (s16Freq.get(t.id) ?? 0) + 1));
      });

      // Game slot tracking
      r.allRoundWinners.forEach((winner, key) => {
        if (!gameSlotFreq.has(key)) gameSlotFreq.set(key, new Map());
        if (!slotWinnerCounts.has(key)) slotWinnerCounts.set(key, new Map());
        const slot = gameSlotFreq.get(key)!;
        slot.set(winner.id, (slot.get(winner.id) ?? 0) + 1);
      });
    }

    done += batch;
    onProgress(done, numSims);
    await new Promise(r => setTimeout(r, 0));
  }

  // Build most-likely bracket by picking argmax winner at each slot
  // Use sharpened model (5x weights) to deterministically pick winners
  const sharpModel: EnsembleModel = {
    ...model,
    lrWeights: model.lrWeights.map(w => w * 6),
    ensembleW: { lr: 0.5, nn: 0.3, elo: 0.15, em: 0.05 },
  };
  const mostLikelyBracket = simulateBracket(regionTeams, sharpModel);

  return {
    championFreq,
    ffFreq,
    e8Freq,
    s16Freq,
    gameSlotFreq,
    totalSims: done,
    mostLikelyBracket,
  };
}

// ─── Display bracket builder ───────────────────────────────────────────────

export interface DisplayGame {
  id: string;
  teamA: Team | null;
  teamB: Team | null;
  winner: Team | null;
  winProbA: number;
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
  consensus: ConsensusData | null,
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
      const slot = consensus?.gameSlotFreq.get(key);
      const total = consensus?.totalSims ?? 1;
      games.push({
        id: key, teamA: a, teamB: b, winner, winProbA: prob,
        round: rd + 1, region: regionName, position: i / 2,
        teamAConsensus: slot ? ((slot.get(a.id) ?? 0) / total) * 100 : undefined,
        teamBConsensus: slot ? ((slot.get(b.id) ?? 0) / total) * 100 : undefined,
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
  consensus: ConsensusData | null,
): DisplayBracket {
  const names = ['East', 'South', 'Midwest', 'West'];
  const [east, south, midwest, west] = names.map((name, idx) =>
    buildRegionGames(regionTeams[idx], name, model, simResult, idx, consensus)
  );

  const [ff0a, ff0b, ff1a, ff1b] = [
    simResult.finalFour[0], simResult.finalFour[1],
    simResult.finalFour[2], simResult.finalFour[3],
  ];

  function ffConsensus(t: Team) {
    return consensus ? ((consensus.ffFreq.get(t.id) ?? 0) / consensus.totalSims) * 100 : undefined;
  }
  function champConsensus(t: Team) {
    return consensus ? ((consensus.championFreq.get(t.id) ?? 0) / consensus.totalSims) * 100 : undefined;
  }

  const champA = simResult.ffWinners[0];
  const champB = simResult.ffWinners[1];

  return {
    east, south, midwest, west,
    finalFour: [
      {
        id: 'ff_0', teamA: ff0a, teamB: ff0b, winner: simResult.ffWinners[0],
        winProbA: ensembleWinProb(model, ff0a, ff0b), round: 5, region: 'Final Four', position: 0,
        teamAConsensus: ffConsensus(ff0a), teamBConsensus: ffConsensus(ff0b),
      },
      {
        id: 'ff_1', teamA: ff1a, teamB: ff1b, winner: simResult.ffWinners[1],
        winProbA: ensembleWinProb(model, ff1a, ff1b), round: 5, region: 'Final Four', position: 1,
        teamAConsensus: ffConsensus(ff1a), teamBConsensus: ffConsensus(ff1b),
      },
    ],
    championship: {
      id: 'champ', teamA: champA, teamB: champB, winner: simResult.champion,
      winProbA: ensembleWinProb(model, champA, champB), round: 6, region: 'Championship', position: 0,
      teamAConsensus: champConsensus(champA), teamBConsensus: champConsensus(champB),
    },
  };
}
