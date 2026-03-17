export interface Team {
  id: string;
  name: string;
  abbreviation?: string;
  seed: number;
  region: 'East' | 'South' | 'Midwest' | 'West';
  espnId: number;
  // KenPom-style adjusted efficiency
  adjOE: number;
  adjDE: number;
  adjTempo: number;
  // Season record
  wins: number;
  losses: number;
  // Quadrant records
  q1W: number; q1L: number;
  q2W: number; q2L: number;
  q3W: number; q3L: number;
  q4W: number; q4L: number;
  // Rankings
  netRanking: number;
  sos: number;
  // Form
  last10: number;
  // Four Factors (offense)
  efgPct: number;
  toRate: number;
  orbPct: number;
  ftRate: number;
  // Four Factors (defense)
  defEfgPct: number;
  defToRate: number;
  defOrbPct: number;
  defFtRate: number;
  // Extra
  threePtRate: number;
  threePtPct: number;
  experience: number;
  coachTourneyWins: number;
  // Player data (populated at runtime from ESPN)
  roster?: Player[];
}

export interface Player {
  id: string;
  name: string;
  jersey: string;
  position: string;   // PG, SG, SF, PF, C
  year: string;       // FR, SO, JR, SR, Grad
  height: string;
  weight: number;
  // Per-game stats
  mpg: number;
  ppg: number;
  rpg: number;
  apg: number;
  spg: number;
  bpg: number;
  topg: number;
  fpg: number;
  // Shooting
  fgPct: number;
  fg3Pct: number;
  ftPct: number;
  fgaPerGame: number;
  fg3aPerGame: number;
  ftaPerGame: number;
  // Advanced
  usageRate: number;      // % of team possessions
  trueShootingPct: number;
  offRtg: number;         // offensive rating (pts per 100 poss when on court)
  defRtg: number;         // defensive rating
  bpm: number;            // box plus/minus
  obpm: number;
  dbpm: number;
  // Clutch / star factor
  isStarter: boolean;
  starScore: number;      // 0-100 composite star rating
}

export interface TeamRoster {
  teamId: string;
  players: Player[];
  fetchedAt: number;
  // Computed team-level player metrics
  starPlayerBPM: number;
  depthScore: number;
  starReliance: number;   // how much team depends on #1 player
  avgExperienceYears: number;
  recruitingRank: number;
}

export interface BracketGame {
  id: string;
  teamA: Team | null;
  teamB: Team | null;
  winner: Team | null;
  winProbA: number;
  marketProbA?: number; // market-implied probability for teamA
  marketProbB?: number; // market-implied probability for teamB
  round: number;
  region: string;
  position: number;
  // Consensus
  teamAConsensus?: number;
  teamBConsensus?: number;
  topTeams?: { team: Team; pct: number }[];
}

export interface SimResult {
  champion: Team;
  finalFour: Team[];
  eliteEight: Team[];
  roundResults: Team[][];
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  lrDecay: number;
  modelType?: 'logistic' | 'neural';
}

export type SimPhase =
  | 'idle'
  | 'fetching'
  | 'generating'
  | 'training-lr'
  | 'training-nn'
  | 'calibrating-elo'
  | 'simulating'
  | 'analyzing'
  | 'done';

export interface SimProgress {
  phase: SimPhase;
  phaseProgress: number;
  overall: number;
  message: string;
  detail: string;
  training?: TrainingProgress;
}

export interface FeatureImportance {
  name: string;
  weight: number;
  absWeight: number;
  rank: number;
}

export interface ModelStats {
  weights: number[];
  finalLoss: number;
  finalAccuracy: number;
  valLoss?: number;
  valAccuracy?: number;
  nnAccuracy?: number;
  featureImportance: FeatureImportance[];
  trainingSamples: number;
  epochs: number;
  ensembleWeights: { lr: number; nn: number; elo: number; em: number; market: number };
}

export type ModelWeights = number[];

export interface NNWeights {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  wOut: number[];
  bOut: number;
}

export interface EloRating {
  teamId: string;
  rating: number;
}
