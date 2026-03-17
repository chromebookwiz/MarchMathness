
// ── Types ──────────────────────────────────────────────────────────────────

export interface CalibrationBin {
  lower: number;
  upper: number;
  predicted: number;
  actual: number;
  count: number;
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

