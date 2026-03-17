'use client';

import type { SimProgress, TrainingProgress } from '@/lib/types';

interface ProgressBarProps {
  progress: SimProgress;
}

const PHASE_LABELS: Record<string, string> = {
  idle: 'Ready',
  generating: 'Generating Season Data',
  training: 'Training Model',
  simulating: 'Running Simulations',
  analyzing: 'Computing Statistics',
  done: 'Analysis Complete',
};

const PHASE_ICONS: Record<string, string> = {
  idle: '●',
  generating: '◈',
  training: '⟳',
  simulating: '▶',
  analyzing: '≡',
  done: '✓',
};

export default function ProgressBar({ progress }: ProgressBarProps) {
  const { phase, overall, message, detail, training } = progress;
  const pct = Math.round(overall * 100);

  return (
    <div className="w-full space-y-3 p-4 rounded-xl border border-[#1a2844] bg-[#080f1f]">
      {/* Phase indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`text-lg font-mono ${phase === 'done' ? 'text-green-400' : 'text-amber-400 animate-spin'}`}
            style={{ display: 'inline-block', animationDuration: '2s' }}
          >
            {PHASE_ICONS[phase] ?? '●'}
          </span>
          <span className="text-sm font-semibold text-slate-200 uppercase tracking-widest">
            {PHASE_LABELS[phase] ?? phase}
          </span>
        </div>
        <span className="text-lg font-bold font-mono text-amber-400">{pct}%</span>
      </div>

      {/* Main bar */}
      <div className="w-full h-3 rounded-full bg-[#0d1a2e] overflow-hidden relative">
        <div
          className="h-full rounded-full transition-all duration-300 ease-out"
          style={{
            width: `${pct}%`,
            background: phase === 'done'
              ? 'linear-gradient(90deg, #22c55e, #4ade80)'
              : 'linear-gradient(90deg, #b45309, #f59e0b, #fbbf24)',
            backgroundSize: '200% auto',
            animation: phase !== 'done' && phase !== 'idle' ? 'progress-shine 2s linear infinite' : 'none',
          }}
        />
        {/* Shimmer overlay */}
        {phase !== 'done' && phase !== 'idle' && (
          <div
            className="absolute inset-0 opacity-30"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%)',
              backgroundSize: '60% 100%',
              animation: 'progress-shine 1.5s linear infinite',
            }}
          />
        )}
      </div>

      {/* Message */}
      <div className="space-y-1">
        <p className="text-sm text-slate-300 font-mono">{message}</p>
        {detail && (
          <p className="text-xs text-slate-500 font-mono">{detail}</p>
        )}
      </div>

      {/* Training details */}
      {training && phase === 'training' && (
        <TrainingDetails training={training} />
      )}
    </div>
  );
}

function TrainingDetails({ training }: { training: TrainingProgress }) {
  const { epoch, totalEpochs, loss, accuracy, lrDecay } = training;
  const trainPct = Math.round((epoch / totalEpochs) * 100);

  return (
    <div className="mt-2 p-3 rounded-lg bg-[#0a1628] border border-[#1a2844]">
      {/* Epoch bar */}
      <div className="flex justify-between text-xs text-slate-500 mb-1 font-mono">
        <span>Epoch {epoch.toLocaleString()} / {totalEpochs.toLocaleString()}</span>
        <span>LR scale: {lrDecay.toFixed(3)}</span>
      </div>
      <div className="w-full h-1.5 rounded-full bg-[#0d1a2e] overflow-hidden mb-3">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-200"
          style={{ width: `${trainPct}%` }}
        />
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-3 gap-3">
        <MetricBox label="Loss" value={loss.toFixed(4)} delta={loss < 0.5 ? '↓' : '→'} />
        <MetricBox label="Accuracy" value={`${(accuracy * 100).toFixed(1)}%`} delta={accuracy > 0.65 ? '↑' : '→'} />
        <MetricBox label="Samples" value="~2.2K" delta="●" />
      </div>
    </div>
  );
}

function MetricBox({ label, value, delta }: { label: string; value: string; delta: string }) {
  return (
    <div className="text-center p-2 rounded bg-[#0d1f3c]">
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-sm font-bold font-mono text-slate-200">{value}</div>
      <div className="text-xs text-amber-400">{delta}</div>
    </div>
  );
}
