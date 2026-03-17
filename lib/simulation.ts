import type { Team, ModelWeights, TrainingProgress, SimProgress, ModelStats, FeatureImportance } from './types';

// ─── Feature Engineering ──────────────────────────────────────────────────────

export const FEATURE_NAMES = [
  'Efficiency Margin Diff',
  'Adj. Offensive Efficiency',
  'Adj. Defensive Efficiency',
  'Adjusted Tempo',
  'NET Ranking',
  'Q1 Win Percentage',
  'Overall Win Percentage',
  'Recent Form (Last 10)',
  'Effective FG% Diff',
  'Turnover Rate Diff',
  'Off. Rebound Rate Diff',
  'Free Throw Rate Diff',
  'Strength of Schedule',
  'Experience Factor',
  'Coach Tournament Pedigree',
  'Defensive Efficiency Score',
  'Scoring Margin (raw)',
  'Q1+Q2 Combined Win%',
];

const NUM_FEATURES = FEATURE_NAMES.length;

function winPct(w: number, l: number) {
  return w / (w + l + 1e-9);
}

function q1WinPct(t: Team) {
  return winPct(t.q1W, t.q1L);
}

function q12WinPct(t: Team) {
  const w = t.q1W + t.q2W;
  const l = t.q1L + t.q2L;
  return winPct(w, l);
}

function defScore(t: Team) {
  // Composite defensive score (higher = better defense)
  return (100 - t.adjDE) * 0.5 + t.defToRate * 0.3 + (100 - t.defEfgPct) * 0.5;
}

function rawScoringMargin(t: Team) {
  return (t.adjOE - t.adjDE) / 30; // normalize
}

export function computeFeatures(a: Team, b: Team): number[] {
  const aEM = a.adjOE - a.adjDE;
  const bEM = b.adjOE - b.adjDE;
  return [
    (aEM - bEM) * 0.08,                                     // 0  EM diff
    (a.adjOE - b.adjOE) * 0.04,                             // 1  OE diff
    (b.adjDE - a.adjDE) * 0.04,                             // 2  DE diff (pos = A better)
    (a.adjTempo - b.adjTempo) * 0.015,                      // 3  Tempo
    (b.netRanking - a.netRanking) * 0.004,                  // 4  NET (pos = A better rank)
    q1WinPct(a) - q1WinPct(b),                              // 5  Q1 win%
    winPct(a.wins, a.losses) - winPct(b.wins, b.losses),    // 6  Overall win%
    (a.last10 - b.last10) / 10,                             // 7  Recent form
    (a.efgPct - b.efgPct) * 0.012,                          // 8  EFG%
    (b.toRate - a.toRate) * 0.06,                           // 9  TO rate (pos = A better)
    (a.orbPct - b.orbPct) * 0.012,                          // 10 ORB%
    (a.ftRate - b.ftRate) * 0.008,                          // 11 FT rate
    (b.sos - a.sos) * 0.003,                                // 12 SOS (pos = A tougher)
    (a.experience - b.experience) * 0.4,                    // 13 Experience
    (a.coachTourneyWins - b.coachTourneyWins) * 0.008,      // 14 Coach pedigree
    (defScore(a) - defScore(b)) * 0.03,                     // 15 Def score
    (rawScoringMargin(a) - rawScoringMargin(b)),             // 16 Scoring margin
    q12WinPct(a) - q12WinPct(b),                            // 17 Q1+Q2 win%
  ];
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function predictWinProb(weights: ModelWeights, a: Team, b: Team): number {
  const features = computeFeatures(a, b);
  const z = features.reduce((s, f, i) => s + weights[i] * f, 0);
  return sigmoid(z);
}

// ─── Training Data Generation ─────────────────────────────────────────────────

interface GameSample {
  features: number[];
  label: number; // 1 = team A won
}

/** Generate a synthetic season game log for all tournament teams.
 *  Each team "plays" ~32 games against synthetic opponents drawn from
 *  quadrant distributions matching their actual Q1-Q4 record. */
export function generateTrainingSamples(teams: Team[]): GameSample[] {
  const samples: GameSample[] = [];

  // Canonical "good weights" used to generate labels — the model will learn to approximate these
  const priorWeights = [1.6, 0.9, 1.0, 0.1, 0.5, 0.8, 0.6, 0.7, 0.4, 0.5, 0.2, 0.1, 0.3, 0.25, 0.15, 0.5, 0.8, 0.7];

  function syntheticOpponent(qualityBand: 1 | 2 | 3 | 4): Team {
    const baseOE = [122, 116, 110, 104][qualityBand - 1] + (Math.random() - 0.5) * 6;
    const baseDE = [93, 101, 107, 114][qualityBand - 1] + (Math.random() - 0.5) * 6;
    const netBase = [20, 55, 115, 200][qualityBand - 1];
    return {
      id: 'opp',
      name: 'Opponent',
      seed: 0,
      region: 'East',
      adjOE: baseOE,
      adjDE: baseDE,
      adjTempo: 67 + (Math.random() - 0.5) * 8,
      wins: 18 + Math.floor(Math.random() * 8),
      losses: 5 + Math.floor(Math.random() * 10),
      q1W: qualityBand === 1 ? 6 : qualityBand === 2 ? 3 : 1,
      q1L: qualityBand === 1 ? 6 : qualityBand === 2 ? 4 : 2,
      q2W: 5, q2L: 3, q3W: 6, q3L: 1, q4W: 4, q4L: 0,
      netRanking: netBase + (Math.random() - 0.5) * 30,
      sos: netBase * 0.8,
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
    // Determine how many games per quadrant (mirrors real schedule composition)
    const netR = team.netRanking;
    const dist: [number, number, number, number] =
      netR <= 15  ? [14, 8, 6, 4] :
      netR <= 35  ? [11, 9, 7, 5] :
      netR <= 65  ? [8,  9, 9, 6] :
                   [4,  7, 11, 10];

    for (let q = 0; q < 4; q++) {
      const band = (q + 1) as 1 | 2 | 3 | 4;
      for (let g = 0; g < dist[q]; g++) {
        const opp = syntheticOpponent(band);
        const features = computeFeatures(team, opp);
        const z = features.reduce((s, f, i) => s + priorWeights[i] * f, 0);
        const prob = sigmoid(z);
        // Add realistic noise to labels
        const noiseProb = Math.max(0.04, Math.min(0.96, prob + (Math.random() - 0.5) * 0.22));
        const won = Math.random() < noiseProb ? 1 : 0;
        samples.push({ features, label: won });
      }
    }

    // Also add direct H2H matchups between tournament teams (within same region strength)
    const peers = teams.filter(t => t.id !== team.id && Math.abs(t.netRanking - team.netRanking) < 15);
    const numH2H = Math.min(4, peers.length);
    for (let i = 0; i < numH2H; i++) {
      const opp = peers[Math.floor(Math.random() * peers.length)];
      const features = computeFeatures(team, opp);
      const z = features.reduce((s, f, i) => s + priorWeights[i] * f, 0);
      const prob = sigmoid(z);
      const noiseProb = Math.max(0.05, Math.min(0.95, prob + (Math.random() - 0.5) * 0.18));
      const won = Math.random() < noiseProb ? 1 : 0;
      samples.push({ features, label: won });
    }
  });

  // Shuffle
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }
  return samples;
}

// ─── Adam Optimizer ───────────────────────────────────────────────────────────

export async function trainModel(
  samples: GameSample[],
  onProgress: (p: TrainingProgress) => void,
): Promise<ModelWeights> {
  const EPOCHS = 2500;
  const LR = 0.015;
  const BETA1 = 0.9;
  const BETA2 = 0.999;
  const EPS = 1e-8;
  const LAMBDA = 0.0008; // L2 regularization

  const weights = Array.from({ length: NUM_FEATURES }, () => (Math.random() - 0.5) * 0.05);
  const m = new Array(NUM_FEATURES).fill(0); // 1st moment
  const v = new Array(NUM_FEATURES).fill(0); // 2nd moment
  let t = 0;

  // Mini-batch size
  const BATCH = 64;
  const n = samples.length;

  let bestLoss = Infinity;
  const lossHistory: number[] = [];

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let totalLoss = 0;
    let correct = 0;

    // Shuffle each epoch
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    // Process mini-batches
    for (let bStart = 0; bStart < n; bStart += BATCH) {
      const bEnd = Math.min(bStart + BATCH, n);
      const grad = new Array(NUM_FEATURES).fill(0);

      for (let si = bStart; si < bEnd; si++) {
        const { features, label } = samples[si];
        const z = features.reduce((s, f, i) => s + weights[i] * f, 0);
        const p = sigmoid(z);
        totalLoss += -(label * Math.log(p + 1e-10) + (1 - label) * Math.log(1 - p + 1e-10));
        if ((p > 0.5) === (label === 1)) correct++;
        const err = p - label;
        for (let fi = 0; fi < NUM_FEATURES; fi++) {
          grad[fi] += err * features[fi];
        }
      }

      const bSize = bEnd - bStart;
      t++;

      // LR schedule: cosine annealing
      const lrSchedule = LR * (0.5 + 0.5 * Math.cos((epoch / EPOCHS) * Math.PI));

      for (let fi = 0; fi < NUM_FEATURES; fi++) {
        const g = grad[fi] / bSize + LAMBDA * weights[fi];
        m[fi] = BETA1 * m[fi] + (1 - BETA1) * g;
        v[fi] = BETA2 * v[fi] + (1 - BETA2) * g * g;
        const mHat = m[fi] / (1 - Math.pow(BETA1, t));
        const vHat = v[fi] / (1 - Math.pow(BETA2, t));
        weights[fi] -= lrSchedule * mHat / (Math.sqrt(vHat) + EPS);
      }
    }

    const avgLoss = totalLoss / n;
    const accuracy = correct / n;
    lossHistory.push(avgLoss);
    if (avgLoss < bestLoss) bestLoss = avgLoss;

    if (epoch % 40 === 0 || epoch === EPOCHS - 1) {
      onProgress({
        epoch: epoch + 1,
        totalEpochs: EPOCHS,
        loss: avgLoss,
        accuracy,
        lrDecay: 0.5 + 0.5 * Math.cos((epoch / EPOCHS) * Math.PI),
      });
      await new Promise(r => setTimeout(r, 0));
    }
  }

  return weights;
}

// ─── Model Statistics ─────────────────────────────────────────────────────────

export function computeModelStats(
  weights: ModelWeights,
  samples: { features: number[]; label: number }[],
  finalLoss: number,
  finalAccuracy: number,
): ModelStats {
  const absWeights = weights.map(Math.abs);
  const maxAbs = Math.max(...absWeights);
  const featureImportance: FeatureImportance[] = weights
    .map((w, i) => ({
      name: FEATURE_NAMES[i],
      weight: w,
      absWeight: Math.abs(w),
      rank: 0,
    }))
    .sort((a, b) => b.absWeight - a.absWeight)
    .map((fi, idx) => ({ ...fi, rank: idx + 1 }));

  return {
    weights,
    finalLoss,
    finalAccuracy,
    featureImportance,
    trainingSamples: samples.length,
    epochs: 2500,
  };
}

// ─── Bracket Simulation ───────────────────────────────────────────────────────

// Standard NCAA bracket seed pairing order within a region:
// Positions pair as: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
// And bracket order is: 1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15
// So within the "top half" of a region: 1,16,8,9 and 5,12,4,13
// "Bottom half": 6,11,3,14 and 7,10,2,15

export interface BracketSimOutput {
  // 4 regions x 4 rounds = 16 arrays, each containing winners
  regionRounds: Team[][][];  // [region][round] = [winners after that round]
  finalFour: Team[];          // 4 region winners
  ffWinners: Team[];          // 2 final four winners
  champion: Team;
  allRoundWinners: Map<string, Team>; // gameKey -> winner
  winProbs: Map<string, number>;      // gameKey -> probability used
}

function simGame(weights: ModelWeights, a: Team, b: Team): { winner: Team; prob: number } {
  const prob = predictWinProb(weights, a, b);
  return { winner: Math.random() < prob ? a : b, prob };
}

export function simulateBracket(
  regionTeams: Team[][], // [0=East,1=South,2=Midwest,3=West], each already in bracket order
  weights: ModelWeights,
): BracketSimOutput {
  const regionRounds: Team[][][] = [[], [], [], []];
  const finalFour: Team[] = [];
  const allRoundWinners = new Map<string, Team>();
  const winProbs = new Map<string, number>();

  // Simulate each region through 4 rounds
  for (let r = 0; r < 4; r++) {
    let pool = [...regionTeams[r]]; // 16 teams in bracket seeding order
    regionRounds[r] = [];

    for (let round = 0; round < 4; round++) {
      const nextPool: Team[] = [];
      const roundWinners: Team[] = [];
      for (let i = 0; i < pool.length; i += 2) {
        const { winner, prob } = simGame(weights, pool[i], pool[i + 1]);
        const key = `r${r}_rd${round}_g${i / 2}`;
        allRoundWinners.set(key, winner);
        winProbs.set(key, winner === pool[i] ? prob : 1 - prob);
        nextPool.push(winner);
        roundWinners.push(winner);
      }
      regionRounds[r].push(roundWinners);
      pool = nextPool;
    }
    finalFour.push(pool[0]);
  }

  // Final Four: East vs South, Midwest vs West
  const ff0 = simGame(weights, finalFour[0], finalFour[1]);
  const ff1 = simGame(weights, finalFour[2], finalFour[3]);
  const ffWinners = [ff0.winner, ff1.winner];

  // Championship
  const champ = simGame(weights, ffWinners[0], ffWinners[1]);

  return {
    regionRounds,
    finalFour,
    ffWinners,
    champion: champ.winner,
    allRoundWinners,
    winProbs,
  };
}

// ─── Consensus Analysis ───────────────────────────────────────────────────────

export interface ConsensusData {
  // Champion frequency
  championFreq: Map<string, number>;
  // Final Four frequency
  ffFreq: Map<string, number>;
  // Elite Eight frequency
  e8Freq: Map<string, number>;
  // For each game slot: how often did each team appear there
  gameSlotFreq: Map<string, Map<string, number>>; // slotKey -> teamId -> count
  totalSims: number;
  // Most likely complete bracket
  mostLikelyBracket: BracketSimOutput;
}

export async function runConsensusAnalysis(
  regionTeams: Team[][],
  weights: ModelWeights,
  numSims: number,
  onProgress: (done: number, total: number) => void,
): Promise<ConsensusData> {
  const championFreq = new Map<string, number>();
  const ffFreq = new Map<string, number>();
  const e8Freq = new Map<string, number>();
  const gameSlotFreq = new Map<string, Map<string, number>>();

  let mostLikelyBracket: BracketSimOutput | null = null;
  let bestChampFreq = -1;

  const BATCH = 100;
  let done = 0;

  while (done < numSims) {
    const batch = Math.min(BATCH, numSims - done);
    for (let i = 0; i < batch; i++) {
      const result = simulateBracket(regionTeams, weights);

      // Track champion
      const cid = result.champion.id;
      championFreq.set(cid, (championFreq.get(cid) ?? 0) + 1);

      // Track Final Four
      result.finalFour.forEach(t => {
        ffFreq.set(t.id, (ffFreq.get(t.id) ?? 0) + 1);
      });

      // Track Elite Eight (round 3 = index 2 of each region = 1 team per region)
      result.regionRounds.forEach(region => {
        const e8 = region[2]; // Sweet 16 winners = Elite 8 participants = region[2]
        if (e8) {
          e8.forEach(t => {
            e8Freq.set(t.id, (e8Freq.get(t.id) ?? 0) + 1);
          });
        }
      });

      // Track game slots
      result.allRoundWinners.forEach((winner, key) => {
        if (!gameSlotFreq.has(key)) gameSlotFreq.set(key, new Map());
        const slot = gameSlotFreq.get(key)!;
        slot.set(winner.id, (slot.get(winner.id) ?? 0) + 1);
      });
    }

    done += batch;
    onProgress(done, numSims);
    await new Promise(r => setTimeout(r, 0));
  }

  // Find most likely bracket (the champion that appeared most often)
  // Re-run bracket with weights set to be deterministic (argmax at each step)
  // We approximate by running a high-weight version
  const deterministicWeights = weights.map(w => w * 5); // sharpen probabilities
  const mostLikely = simulateBracket(regionTeams, deterministicWeights);

  return {
    championFreq,
    ffFreq,
    e8Freq,
    gameSlotFreq,
    totalSims: numSims,
    mostLikelyBracket: mostLikely,
  };
}

// ─── Bracket State Builder ────────────────────────────────────────────────────

export interface DisplayGame {
  id: string;
  teamA: Team | null;
  teamB: Team | null;
  winner: Team | null;
  winProbA: number;
  round: number;
  region: string;
  position: number;
  // Consensus overlay
  teamAConsensus?: number;
  teamBConsensus?: number;
}

export interface DisplayBracket {
  east: DisplayGame[][];   // [round 0..3][game]
  south: DisplayGame[][];
  midwest: DisplayGame[][];
  west: DisplayGame[][];
  finalFour: DisplayGame[];     // [0] = East vs South, [1] = Midwest vs West
  championship: DisplayGame;
}

function buildRegionGames(
  regionTeams: Team[],
  regionName: string,
  weights: ModelWeights,
  simResult: BracketSimOutput,
  regionIdx: number,
  consensus: ConsensusData | null,
  totalSims: number,
): DisplayGame[][] {
  const rounds: DisplayGame[][] = [];

  let pool = [...regionTeams];
  const pools: Team[][] = [pool];

  for (let rd = 0; rd < 3; rd++) {
    const next: Team[] = [];
    for (let i = 0; i < pool.length; i += 2) {
      const key = `r${regionIdx}_rd${rd}_g${i / 2}`;
      const w = simResult.allRoundWinners.get(key);
      next.push(w ?? pool[i]);
    }
    pool = next;
    pools.push([...pool]);
  }

  for (let rd = 0; rd < 4; rd++) {
    const pool0 = pools[rd];
    const games: DisplayGame[] = [];
    for (let i = 0; i < pool0.length; i += 2) {
      const a = pool0[i];
      const b = pool0[i + 1];
      const key = `r${regionIdx}_rd${rd}_g${i / 2}`;
      const winner = simResult.allRoundWinners.get(key) ?? null;
      const prob = predictWinProb(weights, a, b);

      let aConsensus: number | undefined;
      let bConsensus: number | undefined;
      if (consensus) {
        const slot = consensus.gameSlotFreq.get(key);
        const total = consensus.totalSims;
        aConsensus = ((slot?.get(a.id) ?? 0) / total) * 100;
        bConsensus = ((slot?.get(b.id) ?? 0) / total) * 100;
      }

      games.push({
        id: key,
        teamA: a,
        teamB: b,
        winner,
        winProbA: prob,
        round: rd + 1,
        region: regionName,
        position: i / 2,
        teamAConsensus: aConsensus,
        teamBConsensus: bConsensus,
      });
    }
    rounds.push(games);
  }

  return rounds;
}

export function buildDisplayBracket(
  regionTeams: Team[][],
  weights: ModelWeights,
  simResult: BracketSimOutput,
  consensus: ConsensusData | null,
): DisplayBracket {
  const regions = ['East', 'South', 'Midwest', 'West'];
  const [east, south, midwest, west] = regions.map((name, idx) =>
    buildRegionGames(regionTeams[idx], name, weights, simResult, idx, consensus, consensus?.totalSims ?? 0)
  );

  const ff0A = simResult.finalFour[0];
  const ff0B = simResult.finalFour[1];
  const ff1A = simResult.finalFour[2];
  const ff1B = simResult.finalFour[3];

  const ffGame0: DisplayGame = {
    id: 'ff_0',
    teamA: ff0A,
    teamB: ff0B,
    winner: simResult.ffWinners[0],
    winProbA: predictWinProb(weights, ff0A, ff0B),
    round: 5,
    region: 'Final Four',
    position: 0,
    teamAConsensus: consensus ? ((consensus.ffFreq.get(ff0A.id) ?? 0) / consensus.totalSims) * 100 : undefined,
    teamBConsensus: consensus ? ((consensus.ffFreq.get(ff0B.id) ?? 0) / consensus.totalSims) * 100 : undefined,
  };

  const ffGame1: DisplayGame = {
    id: 'ff_1',
    teamA: ff1A,
    teamB: ff1B,
    winner: simResult.ffWinners[1],
    winProbA: predictWinProb(weights, ff1A, ff1B),
    round: 5,
    region: 'Final Four',
    position: 1,
    teamAConsensus: consensus ? ((consensus.ffFreq.get(ff1A.id) ?? 0) / consensus.totalSims) * 100 : undefined,
    teamBConsensus: consensus ? ((consensus.ffFreq.get(ff1B.id) ?? 0) / consensus.totalSims) * 100 : undefined,
  };

  const champA = simResult.ffWinners[0];
  const champB = simResult.ffWinners[1];

  const championship: DisplayGame = {
    id: 'champ',
    teamA: champA,
    teamB: champB,
    winner: simResult.champion,
    winProbA: predictWinProb(weights, champA, champB),
    round: 6,
    region: 'Championship',
    position: 0,
    teamAConsensus: consensus ? ((consensus.championFreq.get(champA.id) ?? 0) / consensus.totalSims) * 100 : undefined,
    teamBConsensus: consensus ? ((consensus.championFreq.get(champB.id) ?? 0) / consensus.totalSims) * 100 : undefined,
  };

  return { east, south, midwest, west, finalFour: [ffGame0, ffGame1], championship };
}
