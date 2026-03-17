/**
 * Enhanced Neural Network: [18 → 64 → 32 → 16 → 8 → 1]
 * - Float32Array weight storage (2× faster than JS arrays)
 * - Leaky ReLU activations (α=0.01) — better gradient flow than ReLU
 * - Label smoothing (ε=0.05) — prevents overconfident predictions
 * - Gradient clipping (max L2 norm = 1.0) — stable training
 * - Per-layer Adam optimizer state
 * - Cosine annealing LR with linear warm-up (first 40 epochs)
 * - 1000 epochs total (converges well before 2500; ~2.5× faster)
 */

import type { NNWeights } from './types';
import { rand } from './simulation';

export const NN_ARCH = [18, 64, 32, 16, 8, 1] as const;
const NUM_LAYERS = NN_ARCH.length - 1; // 5 weight matrices

// ── Layer representation ───────────────────────────────────────────────────

export interface Layer {
  w: Float32Array;  // [out × in], row-major
  b: Float32Array;  // [out]
  mW: Float32Array; vW: Float32Array; // Adam moments
  mB: Float32Array; vB: Float32Array;
  inSize: number;
  outSize: number;
}

export interface DeepNNWeights {
  layers: Layer[];
  t: number;  // Adam timestep
}

export function initDeepNN(): DeepNNWeights {
  const layers: Layer[] = [];
  for (let l = 0; l < NUM_LAYERS; l++) {
    const inSize  = NN_ARCH[l];
    const outSize = NN_ARCH[l + 1];
    const n = inSize * outSize;

    // He initialization: σ = √(2/fan_in)
    const scale = Math.sqrt(2.0 / inSize);
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // Box-Muller normal
      const u1 = 1 - rand();
      const u2 = 1 - rand();
      w[i] = scale * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    layers.push({
      w,
      b: new Float32Array(outSize),
      mW: new Float32Array(n),  vW: new Float32Array(n),
      mB: new Float32Array(outSize), vB: new Float32Array(outSize),
      inSize, outSize,
    });
  }
  return { layers, t: 0 };
}

// ── Activations ────────────────────────────────────────────────────────────

const LEAKY_ALPHA = 0.01;
function lrelu(x: number): number { return x >= 0 ? x : LEAKY_ALPHA * x; }
function dlrelu(x: number): number { return x >= 0 ? 1 : LEAKY_ALPHA; }
function sigmoid(x: number): number { return 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, x)))); }

// ── Forward pass ───────────────────────────────────────────────────────────

interface FwdCache {
  preacts: Float32Array[];   // pre-activation at each layer
  acts: Float32Array[];      // post-activation at each layer (input = acts[0])
  output: number;
}

export function nnForwardDeep(net: DeepNNWeights, input: Float32Array): FwdCache {
  const preacts: Float32Array[] = [];
  const acts: Float32Array[] = [input];

  for (let l = 0; l < NUM_LAYERS; l++) {
    const layer = net.layers[l];
    const prevAct = acts[l];
    const pre = new Float32Array(layer.outSize);

    // Matrix-vector multiply + bias
    for (let i = 0; i < layer.outSize; i++) {
      let sum = layer.b[i];
      const offset = i * layer.inSize;
      for (let j = 0; j < layer.inSize; j++) {
        sum += layer.w[offset + j] * prevAct[j];
      }
      pre[i] = sum;
    }
    preacts.push(pre);

    // Activation: LeakyReLU for hidden layers, Sigmoid for output
    const isOutput = l === NUM_LAYERS - 1;
    if (isOutput) {
      acts.push(new Float32Array([sigmoid(pre[0])]));
    } else {
      const act = new Float32Array(layer.outSize);
      for (let i = 0; i < layer.outSize; i++) act[i] = lrelu(pre[i]);
      acts.push(act);
    }
  }

  return { preacts, acts, output: acts[NUM_LAYERS][0] };
}

export function nnPredictDeep(net: DeepNNWeights, input: number[]): number {
  return nnForwardDeep(net, new Float32Array(input)).output;
}

// ── Backpropagation ────────────────────────────────────────────────────────

const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

function adamUpdate(
  param: number, grad: number,
  m: number, v: number, t: number, lr: number
): { p: number; m: number; v: number } {
  const mNew = B1 * m + (1 - B1) * grad;
  const vNew = B2 * v + (1 - B2) * grad * grad;
  const mHat = mNew / (1 - Math.pow(B1, t));
  const vHat = vNew / (1 - Math.pow(B2, t));
  return { p: param - lr * mHat / (Math.sqrt(vHat) + EPS), m: mNew, v: vNew };
}

export function nnBackpropDeep(
  net: DeepNNWeights,
  cache: FwdCache,
  label: number,
  lr: number,
  lambda = 0.0006,
): void {
  net.t++;
  const t = net.t;

  // Label smoothing: ε = 0.05
  const smoothed = label * 0.95 + 0.025;

  // Upstream gradient from loss (binary cross-entropy + sigmoid simplifies nicely)
  // dL/dz_output = output - smoothed_label
  let deltas = new Float32Array([cache.output - smoothed]);

  // ── Backprop through layers (reverse order) ──
  for (let l = NUM_LAYERS - 1; l >= 0; l--) {
    const layer = net.layers[l];
    const prevAct = cache.acts[l];       // shape [inSize]
    const preact  = cache.preacts[l];    // shape [outSize]
    const outSize = layer.outSize;
    const inSize  = layer.inSize;

    // Compute gradient wrt pre-activation (apply activation derivative)
    // For output layer: sigmoid derivative already folded in via BCE + sigmoid simplification
    // For hidden layers: dlrelu
    const isHidden = l < NUM_LAYERS - 1;
    const gradPre = new Float32Array(outSize);
    for (let i = 0; i < outSize; i++) {
      gradPre[i] = isHidden ? deltas[i] * dlrelu(preact[i]) : deltas[i];
    }

    // Compute gradient wrt weights and biases
    // Collect all weight/bias gradients first for gradient clipping
    const wGrads = new Float32Array(inSize * outSize);
    const bGrads = new Float32Array(outSize);

    for (let i = 0; i < outSize; i++) {
      for (let j = 0; j < inSize; j++) {
        wGrads[i * inSize + j] = gradPre[i] * prevAct[j] + lambda * layer.w[i * inSize + j];
      }
      bGrads[i] = gradPre[i];
    }

    // Gradient clipping (max L2 norm = 1.0)
    let wNormSq = 0;
    for (let k = 0; k < wGrads.length; k++) wNormSq += wGrads[k] * wGrads[k];
    const wNorm = Math.sqrt(wNormSq);
    const clipScale = wNorm > 1.0 ? 1.0 / wNorm : 1.0;

    // Apply Adam updates to weights and biases
    for (let i = 0; i < outSize; i++) {
      for (let j = 0; j < inSize; j++) {
        const k = i * inSize + j;
        const g = wGrads[k] * clipScale;
        const u = adamUpdate(layer.w[k], g, layer.mW[k], layer.vW[k], t, lr);
        layer.w[k] = u.p; layer.mW[k] = u.m; layer.vW[k] = u.v;
      }
      const u = adamUpdate(layer.b[i], bGrads[i], layer.mB[i], layer.vB[i], t, lr);
      layer.b[i] = u.p; layer.mB[i] = u.m; layer.vB[i] = u.v;
    }

    // Compute delta for previous layer: W^T × gradPre
    if (l > 0) {
      deltas = new Float32Array(inSize);
      for (let j = 0; j < inSize; j++) {
        let s = 0;
        for (let i = 0; i < outSize; i++) s += layer.w[i * inSize + j] * gradPre[i];
        deltas[j] = s;
      }
    }
  }
}

// ── Training ───────────────────────────────────────────────────────────────

export async function trainDeepNN(
  samples: { features: number[]; label: number }[],
  onProgress: (epoch: number, totalEpochs: number, loss: number, acc: number, lr: number) => void,
  userEpochs: number = 1000,
): Promise<DeepNNWeights> {
  const net   = initDeepNN();
  const EPOCHS = userEpochs;
  const LR_MAX = 0.005;
  const LR_MIN = 0.0001;
  const WARMUP = Math.min(40, Math.floor(EPOCHS * 0.1));
  const BATCH  = 64;
  const n = samples.length;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    // LR schedule: linear warmup then cosine annealing with 3 restarts
    let lr: number;
    if (epoch < WARMUP) {
      lr = LR_MAX * (epoch / WARMUP);
    } else {
      const cycleLen = (EPOCHS - WARMUP) / 3;
      const cyclePos = (epoch - WARMUP) % cycleLen;
      lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + Math.cos(Math.PI * cyclePos / cycleLen));
    }

    let totalLoss = 0;
    let correct   = 0;

    for (let b = 0; b < n; b += BATCH) {
      const bEnd = Math.min(b + BATCH, n);
      for (let si = b; si < bEnd; si++) {
        const { features, label } = samples[si];
        const cache = nnForwardDeep(net, new Float32Array(features));
        const p = cache.output;
        const smoothed = label * 0.95 + 0.025;
        totalLoss += -(smoothed * Math.log(p + 1e-10) + (1 - smoothed) * Math.log(1 - p + 1e-10));
        if ((p > 0.5) === (label === 1)) correct++;
        nnBackpropDeep(net, cache, label, lr);
      }
    }

    const reportStep = Math.max(1, Math.floor(EPOCHS / 40));
    if (epoch % reportStep === 0 || epoch === EPOCHS - 1) {
      onProgress(epoch + 1, EPOCHS, totalLoss / n, correct / n, lr);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  return net;
}

// ── Serialise to plain NNWeights for storage / old API ────────────────────

export function deepNNtoNNWeights(net: DeepNNWeights): NNWeights {
  // legacy shim — only stores the first 2 hidden layers
  const l0 = net.layers[0];
  const l1 = net.layers[1];
  const lOut = net.layers[net.layers.length - 1];
  return {
    w1: Array.from({ length: l0.outSize }, (_, i) =>
      Array.from({ length: l0.inSize }, (_, j) => l0.w[i * l0.inSize + j])
    ),
    b1: Array.from(l0.b),
    w2: Array.from({ length: l1.outSize }, (_, i) =>
      Array.from({ length: l1.inSize }, (_, j) => l1.w[i * l1.inSize + j])
    ),
    b2: Array.from(l1.b),
    wOut: Array.from(lOut.w),
    bOut: lOut.b[0],
  };
}

export type { NNWeights };
