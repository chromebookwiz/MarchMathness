import type { NNWeights } from './types';

// ── 2-layer MLP with ReLU hidden layers, sigmoid output ───────────────────
// Architecture: [18] → [32, ReLU] → [16, ReLU] → [1, Sigmoid]

const INPUT  = 18;
const HIDDEN1 = 32;
const HIDDEN2 = 16;
const OUTPUT  = 1;

function rand(scale: number) {
  return (Math.random() * 2 - 1) * scale;
}

export function initNN(): NNWeights {
  // He initialization for ReLU layers
  const s1 = Math.sqrt(2 / INPUT);
  const s2 = Math.sqrt(2 / HIDDEN1);
  const s3 = Math.sqrt(2 / HIDDEN2);

  const w1 = Array.from({ length: HIDDEN1 }, () =>
    Array.from({ length: INPUT }, () => rand(s1))
  );
  const b1 = new Array(HIDDEN1).fill(0);

  const w2 = Array.from({ length: HIDDEN2 }, () =>
    Array.from({ length: HIDDEN1 }, () => rand(s2))
  );
  const b2 = new Array(HIDDEN2).fill(0);

  const wOut = Array.from({ length: HIDDEN2 }, () => rand(s3));
  const bOut = 0;

  return { w1, b1, w2, b2, wOut, bOut };
}

function relu(x: number) { return x > 0 ? x : 0; }
function sigmoid(x: number) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }
function drelu(x: number) { return x > 0 ? 1 : 0; }

export interface NNCache {
  x: number[];
  h1: number[];   // pre-activation
  a1: number[];   // post-activation
  h2: number[];
  a2: number[];
  z: number;
  output: number;
}

export function nnForward(net: NNWeights, x: number[]): NNCache {
  // Layer 1
  const h1 = net.w1.map((row, i) =>
    row.reduce((s, w, j) => s + w * x[j], 0) + net.b1[i]
  );
  const a1 = h1.map(relu);

  // Layer 2
  const h2 = net.w2.map((row, i) =>
    row.reduce((s, w, j) => s + w * a1[j], 0) + net.b2[i]
  );
  const a2 = h2.map(relu);

  // Output
  const z = net.wOut.reduce((s, w, j) => s + w * a2[j], 0) + net.bOut;
  const output = sigmoid(z);

  return { x, h1, a1, h2, a2, z, output };
}

export function nnPredict(net: NNWeights, x: number[]): number {
  return nnForward(net, x).output;
}

// ── Adam optimizer state for NN ───────────────────────────────────────────

export interface AdamStateNN {
  mW1: number[][]; vW1: number[][];
  mB1: number[];   vB1: number[];
  mW2: number[][]; vW2: number[][];
  mB2: number[];   vB2: number[];
  mWOut: number[]; vWOut: number[];
  mBOut: number;   vBOut: number;
  t: number;
}

export function initAdamNN(): AdamStateNN {
  const zeroLike = (arr: number[]) => new Array(arr.length).fill(0);
  const zeroLike2 = (arr: number[][]) => arr.map(r => new Array(r.length).fill(0));
  // Dummy init — will be replaced on first forward
  const dummy: NNWeights = initNN();
  return {
    mW1: zeroLike2(dummy.w1), vW1: zeroLike2(dummy.w1),
    mB1: zeroLike(dummy.b1), vB1: zeroLike(dummy.b1),
    mW2: zeroLike2(dummy.w2), vW2: zeroLike2(dummy.w2),
    mB2: zeroLike(dummy.b2), vB2: zeroLike(dummy.b2),
    mWOut: zeroLike(dummy.wOut), vWOut: zeroLike(dummy.wOut),
    mBOut: 0, vBOut: 0,
    t: 0,
  };
}

function adamUpdate(
  param: number,
  grad: number,
  m: number,
  v: number,
  t: number,
  lr: number,
  b1 = 0.9, b2 = 0.999, eps = 1e-8
): { param: number; m: number; v: number } {
  const mNew = b1 * m + (1 - b1) * grad;
  const vNew = b2 * v + (1 - b2) * grad * grad;
  const mHat = mNew / (1 - Math.pow(b1, t));
  const vHat = vNew / (1 - Math.pow(b2, t));
  return { param: param - lr * mHat / (Math.sqrt(vHat) + eps), m: mNew, v: vNew };
}

// ── Backpropagation + Adam update ─────────────────────────────────────────

export function nnBackprop(
  net: NNWeights,
  adam: AdamStateNN,
  cache: NNCache,
  label: number,
  lr: number,
  lambda = 0.001,
): void {
  adam.t++;
  const t = adam.t;

  // Output gradient
  const dz = cache.output - label;

  // Gradients wOut
  for (let j = 0; j < HIDDEN2; j++) {
    const g = dz * cache.a2[j] + lambda * net.wOut[j];
    const u = adamUpdate(net.wOut[j], g, adam.mWOut[j], adam.vWOut[j], t, lr);
    net.wOut[j] = u.param; adam.mWOut[j] = u.m; adam.vWOut[j] = u.v;
  }
  // bOut
  {
    const u = adamUpdate(net.bOut, dz, adam.mBOut, adam.vBOut, t, lr);
    net.bOut = u.param; adam.mBOut = u.m; adam.vBOut = u.v;
  }

  // Gradient into a2
  const da2 = net.wOut.map(w => dz * w);

  // Through ReLU of layer 2
  const dh2 = da2.map((d, i) => d * drelu(cache.h2[i]));

  // Gradients w2, b2
  for (let i = 0; i < HIDDEN2; i++) {
    for (let j = 0; j < HIDDEN1; j++) {
      const g = dh2[i] * cache.a1[j] + lambda * net.w2[i][j];
      const u = adamUpdate(net.w2[i][j], g, adam.mW2[i][j], adam.vW2[i][j], t, lr);
      net.w2[i][j] = u.param; adam.mW2[i][j] = u.m; adam.vW2[i][j] = u.v;
    }
    const u = adamUpdate(net.b2[i], dh2[i], adam.mB2[i], adam.vB2[i], t, lr);
    net.b2[i] = u.param; adam.mB2[i] = u.m; adam.vB2[i] = u.v;
  }

  // Gradient into a1
  const da1 = new Array(HIDDEN1).fill(0);
  for (let j = 0; j < HIDDEN1; j++) {
    for (let i = 0; i < HIDDEN2; i++) {
      da1[j] += dh2[i] * net.w2[i][j];
    }
  }

  // Through ReLU of layer 1
  const dh1 = da1.map((d, i) => d * drelu(cache.h1[i]));

  // Gradients w1, b1
  for (let i = 0; i < HIDDEN1; i++) {
    for (let j = 0; j < INPUT; j++) {
      const g = dh1[i] * cache.x[j] + lambda * net.w1[i][j];
      const u = adamUpdate(net.w1[i][j], g, adam.mW1[i][j], adam.vW1[i][j], t, lr);
      net.w1[i][j] = u.param; adam.mW1[i][j] = u.m; adam.vW1[i][j] = u.v;
    }
    const u = adamUpdate(net.b1[i], dh1[i], adam.mB1[i], adam.vB1[i], t, lr);
    net.b1[i] = u.param; adam.mB1[i] = u.m; adam.vB1[i] = u.v;
  }
}

export async function trainNeuralNet(
  samples: { features: number[]; label: number }[],
  onProgress: (epoch: number, loss: number, acc: number) => void,
): Promise<NNWeights> {
  const net  = initNN();
  const adam = initAdamNN();
  const EPOCHS = 1200;
  const LR_MAX = 0.003;
  const n = samples.length;
  const BATCH = 48;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    let totalLoss = 0;
    let correct = 0;
    // Cosine annealing with warm restarts every 400 epochs
    const cycle = epoch % 400;
    const lr = LR_MAX * (0.5 + 0.5 * Math.cos((cycle / 400) * Math.PI));

    for (let b = 0; b < n; b += BATCH) {
      const bEnd = Math.min(b + BATCH, n);
      for (let si = b; si < bEnd; si++) {
        const { features, label } = samples[si];
        const cache = nnForward(net, features);
        const p = cache.output;
        totalLoss += -(label * Math.log(p + 1e-10) + (1 - label) * Math.log(1 - p + 1e-10));
        if ((p > 0.5) === (label === 1)) correct++;
        nnBackprop(net, adam, cache, label, lr);
      }
    }

    if (epoch % 60 === 0 || epoch === EPOCHS - 1) {
      onProgress(epoch + 1, totalLoss / n, correct / n);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  return net;
}
