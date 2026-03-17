'use client';

import type { SimProgress, TrainingProgress } from '@/lib/types';

const PHASE_LABELS: Record<string, string> = {
  idle:           'STANDBY',
  fetching:       'FETCHING PLAYER DATA',
  generating:     'GENERATING SEASON LOGS',
  'training-lr':  'TRAINING — LOGISTIC REGRESSION',
  'training-nn':  'TRAINING — NEURAL NETWORK',
  'calibrating-elo': 'CALIBRATING ELO RATINGS',
  simulating:     'RUNNING SIMULATIONS',
  analyzing:      'AGGREGATING RESULTS',
  done:           'ANALYSIS COMPLETE',
};

export default function ProgressBar({ progress }: { progress: SimProgress }) {
  const { phase, overall, message, detail, training } = progress;
  const pct = Math.round(overall * 100);
  const isDone = phase === 'done';

  return (
    <div className="border border-[#1e3a5f] bg-[#040d1a]" style={{ fontFamily: 'monospace' }}>
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-[#1e3a5f] px-3 py-1.5">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2"
            style={{
              background: isDone ? '#22c55e' : '#f59e0b',
              animation: isDone ? 'none' : 'pulse 1s ease-in-out infinite',
            }}
          />
          <span className="font-bold">
            {PHASE_LABELS[phase] ?? phase}
          </span>
        </div>
        <span className="font-black tabular-nums">{pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="px-3 py-2">
        <div className="w-full h-2 bg-[#0a1628] relative overflow-hidden">
          <div
            className="h-full transition-all duration-300"
            style={{
              width: `${pct}%`,
              background: isDone
                ? '#22c55e'
                : `linear-gradient(90deg, #92400e 0%, #f59e0b ${pct > 20 ? '60%' : '100%'}, #fbbf24 100%)`,
            }}
          />
          {/* Tick marks */}
          {[25, 50, 75].map(tick => (
            <div
              key={tick}
              className="absolute top-0 bottom-0 w-px bg-[#1e3a5f]"
              style={{ left: `${tick}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-0.5">
          {[0, 25, 50, 75, 100].map(tick => (
            <span key={tick} className="tabular-nums">{tick}</span>
          ))}
        </div>
      </div>

      {/* Message */}
      <div className="px-3 pb-2 space-y-0.5">
        <div>{message}</div>
        {detail && <div>{detail}</div>}
      </div>

      {/* Training metrics */}
      {training && (phase === 'training-lr' || phase === 'training-nn') && (
        <TrainingPanel training={training} />
      )}
    </div>
  );
}

function TrainingPanel({ training }: { training: TrainingProgress }) {
  const { epoch, totalEpochs, loss, accuracy, lrDecay, modelType } = training;
  const trainPct = Math.round((epoch / totalEpochs) * 100);

  return (
    <div className="border-t border-[#1e3a5f] mx-3 mb-3 pt-2">
      <div className="grid grid-cols-4 gap-1 mb-2">
        <MetricCell label="EPOCH" value={`${epoch}/${totalEpochs}`} />
        <MetricCell label="LOSS" value={loss.toFixed(4)} highlight={loss < 0.45} />
        <MetricCell label="ACCURACY" value={`${(accuracy * 100).toFixed(1)}%`} highlight={accuracy > 0.65} />
        <MetricCell label="LR SCALE" value={lrDecay.toFixed(3)} />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[9px] text-slate-700 uppercase tracking-widest w-16">
          {modelType === 'neural' ? 'NN' : 'LR'} EPOCH
        </span>
        <div className="flex-1 h-1 bg-[#0a1628] relative">
          <div
            className="h-full transition-all duration-100"
            style={{ width: `${trainPct}%`, background: modelType === 'neural' ? '#3b82f6' : '#f59e0b' }}
          />
        </div>
        <span className="text-[9px] text-slate-600 tabular-nums w-8">{trainPct}%</span>
      </div>
    </div>
  );
}

function MetricCell({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="bg-[#08111e] px-2 py-1 border border-[#0d1e30]">
      <div className="text-[8px] text-slate-600 uppercase tracking-widest">{label}</div>
      <div
        className="text-xs font-bold tabular-nums mt-0.5"
        style={{ color: highlight ? '#22c55e' : '#94a3b8' }}
      >
        {value}
      </div>
    </div>
  );
}
