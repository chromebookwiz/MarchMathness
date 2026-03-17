import type { Team } from './types';
import type { ConsensusData } from './simulation';

// ── Types ──────────────────────────────────────────────────────────────────

export interface CalibrationBin {
  lower: number;
  upper: number;
  predicted: number;
  actual: number;
  count: number;
}

export interface TeamOdds {
  team: Team;
  champPct: number;
  f4Pct: number;
  e8Pct: number;
  s16Pct: number;
  champCI: { lower: number; upper: number };
  expectedWins: number;
}

export interface BracketStats {
  brierScore: number;
  logLoss: number;
  ece: number;
  calibrationCurve: CalibrationBin[];
  teamOdds: TeamOdds[];
  seedUpsetFreq: Map<string, number>;
}

// ── Core statistics ────────────────────────────────────────────────────────

export function brierScore(predictions: { prob: number; outcome: number }[]): number {
  if (!predictions.length) return 0;
  return predictions.reduce((s, { prob, outcome }) => s + (prob - outcome) ** 2, 0) / predictions.length;
}

export function computeECE(
  predictions: { prob: number; outcome: number }[],
  numBins = 10,
): number {
  const bins = Array.from({ length: numBins }, () => ({ sumProb: 0, sumOut: 0, count: 0 }));
  for (const { prob, outcome } of predictions) {
    const i = Math.min(numBins - 1, Math.floor(prob * numBins));
    bins[i].sumProb += prob;
    bins[i].sumOut  += outcome;
    bins[i].count++;
  }
  const n = predictions.length || 1;
  return bins.reduce((ece, b) => {
    if (!b.count) return ece;
    return ece + (b.count / n) * Math.abs(b.sumProb / b.count - b.sumOut / b.count);
  }, 0);
}

export function calibrationCurve(
  predictions: { prob: number; outcome: number }[],
  numBins = 10,
): CalibrationBin[] {
  const step = 1 / numBins;
  const bins: CalibrationBin[] = Array.from({ length: numBins }, (_, i) => ({
    lower: i * step,
    upper: (i + 1) * step,
    predicted: i * step + step / 2,
    actual: 0,
    count: 0,
  }));
  for (const { prob, outcome } of predictions) {
    const i = Math.min(numBins - 1, Math.floor(prob * numBins));
    bins[i].predicted = (bins[i].predicted * bins[i].count + prob) / (bins[i].count + 1);
    bins[i].actual    = (bins[i].actual    * bins[i].count + outcome) / (bins[i].count + 1);
    bins[i].count++;
  }
  return bins;
}

export function wilsonCI(k: number, n: number, z = 1.96): { lower: number; upper: number } {
  if (n === 0) return { lower: 0, upper: 1 };
  const p      = k / n;
  const z2n    = z * z / n;
  const center = (p + z2n / 2) / (1 + z2n);
  const margin = (z / (1 + z2n)) * Math.sqrt(p * (1 - p) / n + z2n / (4 * n));
  return { lower: Math.max(0, center - margin), upper: Math.min(1, center + margin) };
}

// ── Full bracket stats ─────────────────────────────────────────────────────

export function computeBracketStats(
  consensus: ConsensusData,
  allTeams: Team[],
): BracketStats {
  const { totalSims, championFreq, ffFreq, e8Freq, s16Freq, gameSlotFreq } = consensus;

  // Team odds with Wilson CIs and expected wins
  const teamOdds: TeamOdds[] = allTeams
    .map(t => {
      const champCnt = championFreq.get(t.id) ?? 0;
      const champPct = champCnt / totalSims;
      const f4Pct    = (ffFreq.get(t.id)  ?? 0) / totalSims;
      const e8Pct    = (e8Freq.get(t.id)  ?? 0) / totalSims;
      const s16Pct   = (s16Freq.get(t.id) ?? 0) / totalSims;

      // E[wins] from round probabilities:
      //   R32 → 1 win (implicit: all R64 starters)
      //   S16 → +1, E8 → +1, F4 → +1, Champ game → +1, Win champ → +1
      const expectedWins = Math.max(0,
        champPct * 6 + (f4Pct - champPct) * 5 + (e8Pct - f4Pct) * 4 + (s16Pct - e8Pct) * 3
      );

      return {
        team: t,
        champPct,
        f4Pct,
        e8Pct,
        s16Pct,
        champCI: wilsonCI(champCnt, totalSims),
        expectedWins,
      };
    })
    .sort((a, b) => b.champPct - a.champPct);

  // Brier score + ECE from game slot consensus frequencies
  // Each slot's winner frequency is the "predicted prob"; we score against the argmax (most likely = outcome)
  const predictions: { prob: number; outcome: number }[] = [];
  for (const [, winners] of gameSlotFreq.entries()) {
    let total = 0;
    for (const cnt of winners.values()) total += cnt;
    if (!total) continue;
    let maxCnt = 0;
    for (const cnt of winners.values()) if (cnt > maxCnt) maxCnt = cnt;
    for (const cnt of winners.values()) {
      const freq    = cnt / total;
      const outcome = cnt === maxCnt ? 1 : 0;
      predictions.push({ prob: freq, outcome });
    }
  }

  const bs       = brierScore(predictions);
  const ece      = computeECE(predictions);
  const calibCrv = calibrationCurve(predictions);

  const logLoss = predictions.length > 0
    ? -predictions.reduce((s, { prob, outcome }) =>
        s + outcome * Math.log(prob + 1e-10) + (1 - outcome) * Math.log(1 - prob + 1e-10), 0
      ) / predictions.length
    : 0;

  // Seed upset frequencies from R1 slot data (slots with rd0 in key)
  const seedUpsetFreq = new Map<string, number>();
  for (const [key, winners] of gameSlotFreq.entries()) {
    if (!key.includes('_rd0_')) continue;
    let total = 0;
    for (const cnt of winners.values()) total += cnt;
    if (!total) continue;
    // We can't easily recover seeds from the slot freq alone without team lookup,
    // so mark participation
    for (const [teamId, cnt] of winners.entries()) {
      const team = allTeams.find(t => t.id === teamId);
      if (team) {
        const k = `seed${team.seed}`;
        seedUpsetFreq.set(k, (seedUpsetFreq.get(k) ?? 0) + cnt / total);
      }
    }
  }

  return { brierScore: bs, logLoss, ece, calibrationCurve: calibCrv, teamOdds, seedUpsetFreq };
}
