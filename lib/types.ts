export interface Team {
  id: string;
  name: string;
  seed: number;
  region: 'East' | 'South' | 'Midwest' | 'West';
  // KenPom-style adjusted efficiency
  adjOE: number;       // Adjusted Offensive Efficiency (pts per 100 possessions)
  adjDE: number;       // Adjusted Defensive Efficiency (lower = better)
  adjTempo: number;    // Adjusted Tempo (possessions per 40 min)
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
  sos: number;         // Strength of schedule (avg opponent NET rank)
  // Form
  last10: number;      // Wins in last 10 games
  // Four Factors (offense)
  efgPct: number;      // Effective Field Goal %
  toRate: number;      // Turnover Rate (turnovers per 100 plays, lower = better)
  orbPct: number;      // Offensive Rebound %
  ftRate: number;      // Free Throw Rate (FTA/FGA)
  // Four Factors (defense / opponent)
  defEfgPct: number;
  defToRate: number;   // Opponent turnover rate forced (higher = better defense)
  defOrbPct: number;   // Opponent ORB% allowed (lower = better defense)
  defFtRate: number;   // Opponent FT rate allowed (lower = better)
  // Extra
  threePtRate: number; // % of shots that are 3-pointers
  threePtPct: number;  // 3-Point %
  experience: number;  // 0-1 (0=all freshmen, 1=all seniors)
  coachTourneyWins: number; // Coach's career NCAA tournament wins
}

export interface BracketGame {
  id: string;
  teamA: Team | null;
  teamB: Team | null;
  winner: Team | null;
  winProbA: number;
  round: number;       // 1=R64, 2=R32, 3=S16, 4=E8, 5=FF, 6=Championship
  region: string;
  position: number;    // 0-indexed within round+region
  // Consensus data
  teamAConsensusPct?: number;
  teamBConsensusPct?: number;
  topTeams?: { team: Team; pct: number }[];
}

export interface BracketState {
  regions: { [key: string]: BracketGame[][] }; // region -> rounds -> games
  finalFour: BracketGame[];
  championship: BracketGame;
  champion: Team | null;
}

export interface SimResult {
  champion: Team;
  finalFour: Team[];
  eliteEight: Team[];
  // round results: round -> [winners]
  roundResults: Team[][];
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  lrDecay: number;
}

export type SimPhase =
  | 'idle'
  | 'generating'
  | 'training'
  | 'simulating'
  | 'analyzing'
  | 'done';

export interface SimProgress {
  phase: SimPhase;
  phaseProgress: number;  // 0-1
  overall: number;        // 0-1
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
  featureImportance: FeatureImportance[];
  trainingSamples: number;
  epochs: number;
}

export type ModelWeights = number[];
