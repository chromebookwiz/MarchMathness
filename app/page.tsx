'use client';

import { useState, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import type { SimProgress, ModelStats } from '@/lib/types';
import type { DisplayBracket, ConsensusData, BracketSimOutput } from '@/lib/simulation';
import type { Team } from '@/lib/types';
import ProgressBar from '@/components/ProgressBar';

// Dynamically import the heavy bracket component to avoid SSR issues
const BracketDisplay = dynamic(() => import('@/components/BracketDisplay'), { ssr: false });

const DEFAULT_CONSENSUS_N = 5000;

type Mode = 'single' | 'consensus';

export default function Home() {
  const [mode, setMode] = useState<Mode>('single');
  const [consensusN, setConsensusN] = useState(DEFAULT_CONSENSUS_N);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SimProgress>({
    phase: 'idle',
    phaseProgress: 0,
    overall: 0,
    message: 'Ready to analyze',
    detail: 'Select a mode and click Generate',
  });
  const [bracket, setBracket] = useState<DisplayBracket | null>(null);
  const [consensus, setConsensus] = useState<ConsensusData | null>(null);
  const [champion, setChampion] = useState<Team | null>(null);
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const abortRef = useRef(false);

  const run = useCallback(async () => {
    if (running) return;
    setRunning(true);
    abortRef.current = false;
    setBracket(null);
    setConsensus(null);
    setChampion(null);

    try {
      // Dynamic imports — keeps initial bundle small
      const { buildRegionTeams } = await import('@/lib/bracket');
      const {
        generateTrainingSamples,
        trainModel,
        computeModelStats,
        simulateBracket,
        runConsensusAnalysis,
        buildDisplayBracket,
      } = await import('@/lib/simulation');

      const regionTeams = buildRegionTeams();

      // ── Phase 1: Generate season data ──────────────────────────────────
      setProgress({
        phase: 'generating',
        phaseProgress: 0,
        overall: 0.02,
        message: 'Generating synthetic season game logs…',
        detail: `Building ~2,200 training samples from 64 teams × ~34 games each`,
      });
      await tick();

      const allTeams = regionTeams.flat();
      const samples = generateTrainingSamples(allTeams);

      setProgress({
        phase: 'generating',
        phaseProgress: 1,
        overall: 0.08,
        message: `Season data ready — ${samples.length.toLocaleString()} labelled matchups`,
        detail: 'Samples span Q1–Q4 opponents with realistic noise and H2H games',
      });
      await tick();

      // ── Phase 2: Train model (gradient descent) ────────────────────────
      setProgress({
        phase: 'training',
        phaseProgress: 0,
        overall: 0.1,
        message: 'Initializing Adam optimizer — 18 features, 2,500 epochs…',
        detail: 'β₁=0.9  β₂=0.999  λ=0.0008  cosine LR annealing',
      });
      await tick();

      let lastLoss = 1;
      let lastAcc = 0;

      const weights = await trainModel(samples, (tp) => {
        lastLoss = tp.loss;
        lastAcc = tp.accuracy;
        const trainFrac = tp.epoch / tp.totalEpochs;
        setProgress({
          phase: 'training',
          phaseProgress: trainFrac,
          overall: 0.1 + trainFrac * 0.45,
          message: `Gradient descent — Epoch ${tp.epoch.toLocaleString()}/${tp.totalEpochs.toLocaleString()}`,
          detail: `Loss: ${tp.loss.toFixed(5)} | Accuracy: ${(tp.accuracy * 100).toFixed(1)}% | LR scale: ${tp.lrDecay.toFixed(3)}`,
          training: tp,
        });
      });

      const stats = computeModelStats(weights, samples, lastLoss, lastAcc);
      setModelStats(stats);

      // ── Phase 3: Simulate bracket(s) ──────────────────────────────────
      if (mode === 'single') {
        setProgress({
          phase: 'simulating',
          phaseProgress: 0,
          overall: 0.6,
          message: 'Simulating tournament bracket…',
          detail: 'Sampling game outcomes from learned probability distributions',
        });
        await tick();

        const simResult = simulateBracket(regionTeams, weights);
        const display = buildDisplayBracket(regionTeams, weights, simResult, null);

        setProgress({
          phase: 'analyzing',
          phaseProgress: 0.5,
          overall: 0.95,
          message: 'Building bracket display…',
          detail: `Predicted champion: ${simResult.champion.name}`,
        });
        await tick();

        setBracket(display);
        setChampion(simResult.champion);
        setProgress({
          phase: 'done',
          phaseProgress: 1,
          overall: 1,
          message: 'Bracket generated!',
          detail: `Champion: ${simResult.champion.name} (${simResult.champion.seed}-seed, ${simResult.champion.region})`,
        });
      } else {
        // Consensus mode
        setProgress({
          phase: 'simulating',
          phaseProgress: 0,
          overall: 0.58,
          message: `Running ${consensusN.toLocaleString()} full tournament simulations…`,
          detail: 'Monte Carlo sampling — each sim independently samples all 63 games',
        });
        await tick();

        const consensusData = await runConsensusAnalysis(
          regionTeams,
          weights,
          consensusN,
          (done, total) => {
            const frac = done / total;
            setProgress({
              phase: 'simulating',
              phaseProgress: frac,
              overall: 0.58 + frac * 0.35,
              message: `Simulating… ${done.toLocaleString()} / ${total.toLocaleString()}`,
              detail: `${(frac * 100).toFixed(1)}% complete — sampling from P(win) distributions`,
            });
          },
        );

        setProgress({
          phase: 'analyzing',
          phaseProgress: 0,
          overall: 0.95,
          message: 'Aggregating results and computing consensus bracket…',
          detail: 'Computing champion/F4/E8 frequencies across all simulations',
        });
        await tick();

        const simResult = consensusData.mostLikelyBracket;
        const display = buildDisplayBracket(regionTeams, weights, simResult, consensusData);

        // Most likely champion
        let bestId = '';
        let bestCount = 0;
        consensusData.championFreq.forEach((cnt, id) => {
          if (cnt > bestCount) { bestCount = cnt; bestId = id; }
        });
        const mostLikelyChamp = allTeams.find(t => t.id === bestId) ?? simResult.champion;

        setBracket(display);
        setConsensus(consensusData);
        setChampion(mostLikelyChamp);

        const champPct = ((bestCount / consensusN) * 100).toFixed(1);
        setProgress({
          phase: 'done',
          phaseProgress: 1,
          overall: 1,
          message: `Consensus analysis complete — ${consensusN.toLocaleString()} simulations`,
          detail: `Most likely champion: ${mostLikelyChamp.name} (${champPct}% of sims)`,
        });
      }
    } catch (err) {
      console.error(err);
      setProgress({
        phase: 'idle',
        phaseProgress: 0,
        overall: 0,
        message: 'Error during simulation',
        detail: String(err),
      });
    } finally {
      setRunning(false);
    }
  }, [running, mode, consensusN]);

  return (
    <main className="min-h-screen flex flex-col" style={{ background: '#020817' }}>
      {/* Header */}
      <header className="border-b border-[#0f1e35] py-6 px-6">
        <div className="max-w-screen-2xl mx-auto">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
            <div className="flex-1 text-center sm:text-left">
              <h1 className="text-3xl font-black tracking-tight" style={{ color: '#f59e0b' }}>
                MARCH <span style={{ color: '#e2e8f0' }}>MATH</span>
                <span style={{ color: '#f59e0b' }}>NESS</span>
              </h1>
              <p className="text-xs text-slate-500 mt-1 font-mono uppercase tracking-widest">
                Logistic Regression · Adam Gradient Descent · Monte Carlo Simulation
              </p>
            </div>
            {modelStats && (
              <ModelBadge stats={modelStats} />
            )}
          </div>
        </div>
      </header>

      {/* Controls */}
      <section className="border-b border-[#0f1e35] py-5 px-6">
        <div className="max-w-screen-2xl mx-auto flex flex-col sm:flex-row items-center gap-4">
          {/* Mode toggle */}
          <div className="flex rounded-lg overflow-hidden border border-[#1a2844] bg-[#08111e]">
            <ModeButton
              active={mode === 'single'}
              onClick={() => setMode('single')}
              label="Generate Bracket"
              sub="1 simulation"
              disabled={running}
            />
            <ModeButton
              active={mode === 'consensus'}
              onClick={() => setMode('consensus')}
              label="Consensus Analysis"
              sub={`${consensusN.toLocaleString()} simulations`}
              disabled={running}
            />
          </div>

          {/* Consensus N slider */}
          {mode === 'consensus' && (
            <div className="flex items-center gap-3 bg-[#08111e] border border-[#1a2844] rounded-lg px-4 py-2">
              <span className="text-xs text-slate-500 uppercase tracking-widest">Sims</span>
              <input
                type="range"
                min={500}
                max={25000}
                step={500}
                value={consensusN}
                onChange={e => setConsensusN(Number(e.target.value))}
                disabled={running}
                className="w-32 accent-amber-500"
              />
              <span className="text-sm font-bold font-mono text-amber-400 w-16">
                {consensusN.toLocaleString()}
              </span>
            </div>
          )}

          {/* Generate button */}
          <button
            onClick={run}
            disabled={running}
            className="relative overflow-hidden font-black text-sm uppercase tracking-widest px-8 py-3 rounded-lg transition-all duration-200"
            style={{
              background: running
                ? '#1a2844'
                : 'linear-gradient(135deg, #b45309, #f59e0b, #fbbf24)',
              color: running ? '#475569' : '#000',
              cursor: running ? 'not-allowed' : 'pointer',
              boxShadow: running ? 'none' : '0 0 20px #f59e0b44',
              letterSpacing: '0.12em',
            }}
          >
            {running ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">⟳</span>
                Analyzing…
              </span>
            ) : (
              mode === 'single' ? 'Generate' : `Run ${consensusN.toLocaleString()} Sims`
            )}
          </button>
        </div>
      </section>

      {/* Progress area */}
      {progress.phase !== 'idle' && (
        <section className="px-6 py-4 border-b border-[#0f1e35]">
          <div className="max-w-3xl mx-auto">
            <ProgressBar progress={progress} />
          </div>
        </section>
      )}

      {/* Champion banner */}
      {champion && !running && (
        <div className="px-6 py-3 flex justify-center">
          <div
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full border-2 text-sm font-bold"
            style={{
              borderColor: '#f59e0b',
              background: '#0d0700',
              boxShadow: '0 0 30px #f59e0b33',
              color: '#fbbf24',
            }}
          >
            <span className="text-xl">🏆</span>
            <span>
              {mode === 'consensus'
                ? `Consensus Champion: `
                : `Predicted Champion: `}
              <span style={{ color: '#fef3c7' }}>
                #{champion.seed} {champion.name}
              </span>
              {mode === 'consensus' && consensus && (
                <span style={{ color: '#f59e0b', marginLeft: 8 }}>
                  ({(((consensus.championFreq.get(champion.id) ?? 0) / consensus.totalSims) * 100).toFixed(1)}% of sims)
                </span>
              )}
            </span>
          </div>
        </div>
      )}

      {/* Model stats (collapsed by default) */}
      {modelStats && !running && (
        <div className="px-6 py-2">
          <details className="max-w-3xl mx-auto">
            <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400 transition-colors font-mono uppercase tracking-widest">
              ▸ Model Statistics &amp; Feature Importance
            </summary>
            <div className="mt-3 p-4 rounded-xl border border-[#1a2844] bg-[#080f1f]">
              <div className="grid grid-cols-3 gap-3 mb-4">
                <Stat label="Training Samples" value={modelStats.trainingSamples.toLocaleString()} />
                <Stat label="Final Loss" value={modelStats.finalLoss.toFixed(5)} />
                <Stat label="Train Accuracy" value={`${(modelStats.finalAccuracy * 100).toFixed(1)}%`} />
              </div>
              <div className="space-y-1.5">
                {modelStats.featureImportance.slice(0, 10).map((fi, i) => (
                  <div key={fi.name} className="flex items-center gap-2">
                    <span className="text-xs text-slate-600 w-4 font-mono">#{i + 1}</span>
                    <div
                      className="h-1.5 rounded-full bg-amber-500 flex-shrink-0"
                      style={{
                        width: `${Math.round((fi.absWeight / modelStats.featureImportance[0].absWeight) * 120)}px`,
                        opacity: 0.4 + 0.6 * (fi.absWeight / modelStats.featureImportance[0].absWeight),
                      }}
                    />
                    <span className="text-xs text-slate-400 flex-1">{fi.name}</span>
                    <span className="text-xs font-mono text-slate-500">{fi.weight.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </details>
        </div>
      )}

      {/* Bracket */}
      {bracket && !running && (
        <section className="flex-1 px-2 py-4">
          <div className="mb-3 px-4">
            <div className="flex items-center gap-2 text-xs text-slate-600 font-mono">
              <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> Winner
              <span className="ml-4 w-2 h-2 rounded-full bg-blue-500 inline-block" /> 1–4 seed
              <span className="ml-2 w-2 h-2 rounded-full bg-green-500 inline-block" /> 5–8 seed
              {mode === 'consensus' && (
                <span className="ml-4 text-amber-500">Percentages = consensus win frequency</span>
              )}
              {mode === 'single' && (
                <span className="ml-4">Percentages = model win probability</span>
              )}
            </div>
          </div>
          <BracketDisplay bracket={bracket} consensus={consensus} champion={champion} />
        </section>
      )}

      {/* Empty state */}
      {!bracket && !running && progress.phase === 'idle' && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-4 max-w-md px-6">
            <div className="text-6xl">🏀</div>
            <h2 className="text-2xl font-black text-slate-400">Ready to Predict</h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              Select <strong className="text-slate-400">Generate Bracket</strong> for a single ML-powered prediction,
              or <strong className="text-slate-400">Consensus Analysis</strong> to run thousands of simulations
              and output the statistically most likely bracket.
            </p>
            <p className="text-xs text-slate-700 font-mono">
              18-feature logistic regression · Adam optimizer · 2,500 epochs · Monte Carlo sampling
            </p>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-[#0f1e35] py-3 px-6 text-center">
        <p className="text-xs text-slate-700 font-mono">
          March MathNess — 2025 NCAA Tournament · ML-powered bracket prediction
        </p>
      </footer>
    </main>
  );
}

function ModeButton({
  active,
  onClick,
  label,
  sub,
  disabled,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  sub: string;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-5 py-2.5 transition-all duration-150"
      style={{
        background: active ? '#0f2040' : 'transparent',
        borderRight: '1px solid #1a2844',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <div
        className="text-sm font-bold"
        style={{ color: active ? '#f59e0b' : '#475569' }}
      >
        {label}
      </div>
      <div className="text-xs font-mono" style={{ color: active ? '#78716c' : '#334155' }}>
        {sub}
      </div>
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center p-2 rounded-lg bg-[#0d1f3c]">
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className="text-base font-bold font-mono text-slate-200">{value}</div>
    </div>
  );
}

function ModelBadge({ stats }: { stats: ModelStats }) {
  return (
    <div className="flex gap-3 text-right">
      <div>
        <div className="text-xs text-slate-600 uppercase tracking-wider">Model Accuracy</div>
        <div className="text-lg font-bold font-mono" style={{ color: '#22c55e' }}>
          {(stats.finalAccuracy * 100).toFixed(1)}%
        </div>
      </div>
      <div>
        <div className="text-xs text-slate-600 uppercase tracking-wider">Loss</div>
        <div className="text-lg font-bold font-mono" style={{ color: '#f59e0b' }}>
          {stats.finalLoss.toFixed(4)}
        </div>
      </div>
    </div>
  );
}

function tick() {
  return new Promise(r => setTimeout(r, 0));
}
