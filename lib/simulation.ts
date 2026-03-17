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
  'Seed Differential',
  '3P% Differential',
  '3P Rate Differential',
  'News Sentiment Diff',
];

export const NUM_FEATURES = FEATURE_NAMES.length; // 19

// ─── Feature engineering ───────────────────────────────────────────────────

function w(a: number, b: number) { return a / (a + b + 1e-9); }
function q1r(t: Team) { return w(t.q1W, t.q1L); }
function q12r(t: Team) { return w(t.q1W + t.q2W, t.q1L + t.q2L); }
function defScore(t: Team) {
  return (110 - t.adjDE) * 0.4 + t.defToRate * 0.35 + (100 - t.defEfgPct) * 0.5;
}

export function computeFeatures(a: Team, b: Team, sentiment?: Map<string, number>): number[] {
  const aEM = a.adjOE - a.adjDE;
  const bEM = b.adjOE - b.adjDE;
  const aSent = sentiment?.get(a.id) ?? 0;
  const bSent = sentiment?.get(b.id) ?? 0;
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
    (a.seed - b.seed) * 0.03,
    (a.threePtPct - b.threePtPct) * 0.5,
    (a.threePtRate - b.threePtRate) * 0.5,
    (aSent - bSent) * 0.5,
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
  sentiment?: Map<string, number>;
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
  const nnP = nnPredictDeep(model.nnWeights, computeFeatures(a, b, model.sentiment));

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

export interface TrainingData {
  train: GameSample[];
  val: GameSample[];
}

export async function generateTrainingSamples(
  historicalSamples: GameSample[],
  trainFrac = 0.8,
): Promise<TrainingData> {
  const shuffled = [...historicalSamples];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const split = Math.floor(shuffled.length * trainFrac);
  return {
    train: shuffled.slice(0, split),
    val: shuffled.slice(split),
  };
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
  valLoss?: number,
  valAccuracy?: number,
): ModelStats {
  const fi: FeatureImportance[] = weights
    .map((w, i) => ({ name: FEATURE_NAMES[i], weight: w, absWeight: Math.abs(w), rank: 0 }))
    .sort((a, b) => b.absWeight - a.absWeight)
    .map((f, idx) => ({ ...f, rank: idx + 1 }));

  return {
    weights,
    finalLoss,
    finalAccuracy,
    valLoss,
    valAccuracy,
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
  matchups?: Map<string, { a: Team; b: Team; probA: number }>;
}

export interface MonteCarloGameStats {
  winners: Map<string, { team: Team; count: number }>;
  total: number;
  underdogWins: number;
}

export interface MonteCarloResult {
  bracket: DisplayBracket;
  stats: Map<string, MonteCarloGameStats>;
  championCounts: Map<string, number>;
}

export function simulateBracket(
  regionTeams: Team[][],
  model: EnsembleModel,
  abortSignal?: { aborted: boolean },
  options?: { randomize?: boolean; randomSeed?: number; temperature?: number },
): BracketSimOutput {
  const { randomize = false, randomSeed, temperature = 0.92 } = options ?? {};
  if (randomize && randomSeed !== undefined) setSeed(randomSeed);

  const regionRounds: Team[][][] = [[], [], [], []];
  const finalFour: Team[] = [];
  const allRoundWinners = new Map<string, Team>();
  const winProbs = new Map<string, number>();

  const pickWinner = (a: Team, b: Team, prob: number): Team => {
    if (!randomize) return prob >= 0.5 ? a : b;
    // Smooth probabilities slightly so underdogs have a chance
    const adjusted = 0.5 + (prob - 0.5) * temperature;
    return rand() < adjusted ? a : b;
  };

  for (let r = 0; r < 4; r++) {
    let pool = [...regionTeams[r]];
    regionRounds[r] = [];

    for (let round = 0; round < 4; round++) {
      const next: Team[] = [];
      for (let i = 0; i < pool.length; i += 2) {
        if (abortSignal?.aborted) return { regionRounds, finalFour: [], ffWinners: [], champion: pool[0], allRoundWinners, winProbs };
        const a = pool[i], b = pool[i + 1];
        const prob = ensembleWinProb(model, a, b);
        const winner = pickWinner(a, b, prob);
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
  const ff0 = pickWinner(finalFour[0], finalFour[1], ff0Prob);
  const ff1 = pickWinner(finalFour[2], finalFour[3], ff1Prob);

  const champProb = ensembleWinProb(model, ff0, ff1);
  const champ = pickWinner(ff0, ff1, champProb);

  return {
    regionRounds,
    finalFour,
    ffWinners: [ff0, ff1],
    champion: champ,
    allRoundWinners,
    winProbs,
  };
}

export interface MonteCarloOptions {
  runs?: number;
  temperature?: number;
  seed?: number;
}

export type MonteCarloProgress = (run: number, total: number) => void;

export async function runMonteCarloBracket(
  regionTeams: Team[][],
  model: EnsembleModel,
  options: MonteCarloOptions = {},
  onProgress?: MonteCarloProgress,
  abortSignal?: { aborted: boolean },
): Promise<MonteCarloResult> {
  const runs = Math.max(1, options.runs ?? 500);
  const temp = options.temperature ?? 0.9;
  const seed = options.seed ?? Date.now();

  const stats = new Map<string, MonteCarloGameStats>();
  const champCounts = new Map<string, number>();

  const addWinner = (key: string, team: Team, isUnderdog: boolean) => {
    let s = stats.get(key);
    if (!s) {
      s = { winners: new Map(), total: 0, underdogWins: 0 };
      stats.set(key, s);
    }
    const prev = s.winners.get(team.id);
    if (prev) prev.count += 1;
    else s.winners.set(team.id, { team, count: 1 });
    s.total += 1;
    if (isUnderdog) s.underdogWins += 1;
  };

  const addChampion = (team: Team) => {
    const prev = champCounts.get(team.id) ?? 0;
    champCounts.set(team.id, prev + 1);
  };

  for (let i = 0; i < runs; i++) {
    if (abortSignal?.aborted) break;
    const runSeed = seed + i * 31;
    const sim = simulateBracket(regionTeams, model, abortSignal, {
      randomize: true,
      randomSeed: runSeed,
      temperature: temp,
    });

    addChampion(sim.champion);

    for (const [key, winner] of sim.allRoundWinners) {
      const winProb = sim.winProbs.get(key) ?? 0.5;
      const isUnderdog = winProb < 0.5;
      addWinner(key, winner, isUnderdog);
    }

    onProgress?.(i + 1, runs);
    // Yield to event loop to keep UI responsive.
    if (i % 50 === 0) await new Promise(r => setTimeout(r, 0));
  }

  const bracket = buildConsensusBracket(regionTeams, model, stats);

  return { bracket, stats, championCounts: champCounts };
}

function buildConsensusBracket(
  regionTeams: Team[][],
  model: EnsembleModel,
  stats: Map<string, MonteCarloGameStats>,
): DisplayBracket {
  const makeDisplay = (regionName: string, regionIdx: number): DisplayGame[][] => {
    const rounds: DisplayGame[][] = [];
    // Round 1 uses fixed bracket teams
    let base = [...regionTeams[regionIdx]];

    for (let rd = 0; rd < 4; rd++) {
      const games: DisplayGame[] = [];
      for (let i = 0; i < base.length; i += 2) {
        const key = `r${regionIdx}_rd${rd}_g${i / 2}`;
        const stat = stats.get(key);

        const topTeams = stat
          ? Array.from(stat.winners.values())
              .sort((a, b) => b.count - a.count)
              .slice(0, 2)
          : [];

        const teamA = topTeams[0]?.team ?? base[i];
        const teamB = topTeams[1]?.team ?? base[i + 1];
        const winner = topTeams[0]?.team ?? base[i];

        const total = stat?.total ?? 1;
        const winA = stat?.winners.get(teamA.id)?.count ?? (teamA.id === winner.id ? total : 0);
        const winB = stat?.winners.get(teamB.id)?.count ?? (teamB.id === winner.id ? total : 0);

        const upsetProb = stat ? (stat.underdogWins / stat.total) : 0;
        const consensusPct = winA / total;

        const prob = consensusPct; // use consensus win probability for display

        games.push({
          id: key,
          teamA,
          teamB,
          winner,
          winProbA: prob,
          marketProbA: undefined,
          marketProbB: undefined,
          round: rd + 1,
          region: regionName,
          position: i / 2,
          upsetProb,
          consensusPct,
          teamAConsensus: winA / total,
          teamBConsensus: winB / total,
          topTeams: topTeams.map(t => ({ team: t.team, pct: t.count / total })),
        });
      }
      rounds.push(games);

      // Build next base line based on consensus winners
      base = games.map(g => g.winner ?? null).filter(Boolean) as Team[];
      if (base.length === 0) break;
    }

    return rounds;
  };

  const east = makeDisplay('East', 0);
  const south = makeDisplay('South', 1);
  const midwest = makeDisplay('Midwest', 2);
  const west = makeDisplay('West', 3);

  // Final Four
  const ff0a = east[3][0].winner!;
  const ff0b = south[3][0].winner!;
  const ff1a = midwest[3][0].winner!;
  const ff1b = west[3][0].winner!;

  const ff0Prob = ensembleWinProb(model, ff0a, ff0b);
  const ff1Prob = ensembleWinProb(model, ff1a, ff1b);
  const ff0Winner = ff0Prob >= 0.5 ? ff0a : ff0b;
  const ff1Winner = ff1Prob >= 0.5 ? ff1a : ff1b;

  const champProb = ensembleWinProb(model, ff0Winner, ff1Winner);
  const champWinner = champProb >= 0.5 ? ff0Winner : ff1Winner;

  const finalFour: DisplayGame[] = [
    {
      id: 'ff_0',
      teamA: ff0a,
      teamB: ff0b,
      winner: ff0Winner,
      winProbA: ff0Prob,
      marketProbA: marketWinProb(ff0a, ff0b),
      marketProbB: 1 - marketWinProb(ff0a, ff0b),
      round: 5,
      region: 'Final Four',
      position: 0,
      upsetProb: 1 - Math.max(ff0Prob, 1 - ff0Prob),
      consensusPct: Math.max(ff0Prob, 1 - ff0Prob),
    },
    {
      id: 'ff_1',
      teamA: ff1a,
      teamB: ff1b,
      winner: ff1Winner,
      winProbA: ff1Prob,
      marketProbA: marketWinProb(ff1a, ff1b),
      marketProbB: 1 - marketWinProb(ff1a, ff1b),
      round: 5,
      region: 'Final Four',
      position: 1,
      upsetProb: 1 - Math.max(ff1Prob, 1 - ff1Prob),
      consensusPct: Math.max(ff1Prob, 1 - ff1Prob),
    },
  ];

  const championship: DisplayGame = {
    id: 'champ',
    teamA: ff0Winner,
    teamB: ff1Winner,
    winner: champWinner,
    winProbA: champProb,
    marketProbA: marketWinProb(ff0Winner, ff1Winner),
    marketProbB: 1 - marketWinProb(ff0Winner, ff1Winner),
    round: 6,
    region: 'Championship',
    position: 0,
    upsetProb: 1 - Math.max(champProb, 1 - champProb),
    consensusPct: Math.max(champProb, 1 - champProb),
  };

  return { east, south, midwest, west, finalFour, championship };
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
  // Monte Carlo consensus / upset metrics
  upsetProb?: number;
  consensusPct?: number;
  teamAConsensus?: number;
  teamBConsensus?: number;
  topTeams?: { team: Team; pct: number }[];
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
