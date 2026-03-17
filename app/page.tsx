'use client';

import { useState, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import type { SimProgress, ModelStats, Team } from '@/lib/types';
import type { DisplayBracket, ConsensusData, EnsembleModel } from '@/lib/simulation';
import ProgressBar from '@/components/ProgressBar';

const BracketDisplay = dynamic(() => import('@/components/BracketDisplay'), { ssr: false });

const DEFAULT_N = 5000;

type Mode = 'single' | 'consensus';

export default function Home() {
  const [mode, setMode]         = useState<Mode>('single');
  const [simN, setSimN]         = useState(DEFAULT_N);
  const [running, setRunning]   = useState(false);
  const [progress, setProgress] = useState<SimProgress>({
    phase: 'idle', phaseProgress: 0, overall: 0,
    message: 'SELECT A MODE AND CLICK GENERATE',
    detail: '',
  });
  const [bracket, setBracket]   = useState<DisplayBracket | null>(null);
  const [consensus, setConsensus] = useState<ConsensusData | null>(null);
  const [champion, setChampion] = useState<Team | null>(null);
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [fetchedTeams, setFetchedTeams] = useState(0);
  const abortRef = useRef({ aborted: false });

  const run = useCallback(async () => {
    if (running) return;
    setRunning(true);
    abortRef.current = { aborted: false };
    setBracket(null); setConsensus(null); setChampion(null);

    try {
      const { buildRegionTeams } = await import('@/lib/bracket');
      const {
        generateTrainingSamples, trainLogisticRegression, computeModelStats,
        simulateBracket, runConsensusAnalysis, buildDisplayBracket,
        buildEloRatings,
      } = await import('@/lib/simulation');
      const { trainNeuralNet, initNN } = await import('@/lib/neuralNet');
      const { fetchAllRosters, computeRosterMetrics } = await import('@/lib/espn');

      const regionTeams = buildRegionTeams();
      const allTeams = regionTeams.flat();

      // ── Phase 1: Fetch ESPN player data ──────────────────────────────
      setProgress({
        phase: 'fetching', phaseProgress: 0, overall: 0.02,
        message: 'FETCHING LIVE PLAYER DATA FROM ESPN API',
        detail: `Requesting rosters for all ${allTeams.length} tournament teams…`,
      });
      await tick();

      let fetchedCount = 0;
      const rosterMap = await fetchAllRosters(
        allTeams.map(t => t.id),
        (done, total) => {
          fetchedCount = done;
          setFetchedTeams(done);
          setProgress({
            phase: 'fetching',
            phaseProgress: done / total,
            overall: 0.02 + (done / total) * 0.12,
            message: `FETCHING ROSTERS — ${done}/${total} TEAMS`,
            detail: `${allTeams[done - 1]?.name ?? ''} — ${(allTeams[done - 1]?.roster?.length ?? 0)} players`,
          });
        },
      );

      // Attach rosters to teams
      for (const team of allTeams) {
        const players = rosterMap.get(team.id) ?? [];
        (team as Team).roster = players;
      }

      const fetchedWithRosters = allTeams.filter(t => (t.roster?.length ?? 0) > 0).length;

      setProgress({
        phase: 'fetching', phaseProgress: 1, overall: 0.14,
        message: `PLAYER DATA LOADED — ${fetchedWithRosters}/${allTeams.length} TEAMS`,
        detail: `Total players: ${allTeams.reduce((s, t) => s + (t.roster?.length ?? 0), 0)} | Using player-adjusted features`,
      });
      await tick();

      // ── Phase 2: Generate training samples ───────────────────────────
      setProgress({
        phase: 'generating', phaseProgress: 0, overall: 0.16,
        message: 'GENERATING SYNTHETIC SEASON GAME LOGS',
        detail: '~2,400 labelled matchup samples • Q1-Q4 opponent distribution',
      });
      await tick();

      const samples = generateTrainingSamples(allTeams);
      // Duplicate for nn training
      const samplesNN = [...samples];

      setProgress({
        phase: 'generating', phaseProgress: 1, overall: 0.20,
        message: `SEASON DATA READY — ${samples.length.toLocaleString()} SAMPLES`,
        detail: 'H2H matchups + quadrant distribution + player-adjusted labels',
      });
      await tick();

      // ── Phase 3: Train LR model ───────────────────────────────────────
      setProgress({
        phase: 'training-lr', phaseProgress: 0, overall: 0.22,
        message: 'TRAINING LOGISTIC REGRESSION — ADAM OPTIMIZER',
        detail: 'β₁=0.9  β₂=0.999  λ=0.0008  cosine LR  2000 epochs  batch=64',
      });
      await tick();

      let lrFinalLoss = 1, lrFinalAcc = 0;
      const lrWeights = await trainLogisticRegression(samples, tp => {
        lrFinalLoss = tp.loss; lrFinalAcc = tp.accuracy;
        setProgress({
          phase: 'training-lr',
          phaseProgress: tp.epoch / tp.totalEpochs,
          overall: 0.22 + (tp.epoch / tp.totalEpochs) * 0.22,
          message: `LR GRADIENT DESCENT — EPOCH ${tp.epoch.toLocaleString()}/${tp.totalEpochs.toLocaleString()}`,
          detail: `Loss: ${tp.loss.toFixed(5)}  Acc: ${(tp.accuracy * 100).toFixed(1)}%  LR: ${tp.lrDecay.toFixed(4)}`,
          training: tp,
        });
      });

      // ── Phase 4: Train Neural Net ─────────────────────────────────────
      setProgress({
        phase: 'training-nn', phaseProgress: 0, overall: 0.45,
        message: 'TRAINING NEURAL NETWORK [18→32→16→1]',
        detail: 'ReLU activations • cosine warm restarts • backpropagation • 1200 epochs',
      });
      await tick();

      let nnFinalAcc = 0;
      const nnWeights = await trainNeuralNet(samplesNN, (epoch, loss, acc) => {
        nnFinalAcc = acc;
        setProgress({
          phase: 'training-nn',
          phaseProgress: epoch / 1200,
          overall: 0.45 + (epoch / 1200) * 0.15,
          message: `NEURAL NET BACKPROP — EPOCH ${epoch}/1200`,
          detail: `Loss: ${loss.toFixed(5)}  Acc: ${(acc * 100).toFixed(1)}%`,
          training: { epoch, totalEpochs: 1200, loss, accuracy: acc, lrDecay: 0.003, modelType: 'neural' },
        });
      });

      // ── Phase 5: Calibrate Elo ────────────────────────────────────────
      setProgress({
        phase: 'calibrating-elo', phaseProgress: 0, overall: 0.61,
        message: 'CALIBRATING ELO RATINGS FROM SEASON PERFORMANCE',
        detail: 'NET rank + efficiency margin + SOS → initial Elo scores',
      });
      await tick();

      const elos = buildEloRatings(allTeams);
      const stats = computeModelStats(lrWeights, samples, lrFinalLoss, lrFinalAcc, nnFinalAcc);
      setModelStats(stats);

      const model: EnsembleModel = {
        lrWeights,
        nnWeights,
        elos,
        ensembleW: { lr: 0.35, nn: 0.35, elo: 0.20, em: 0.10 },
      };

      setProgress({
        phase: 'calibrating-elo', phaseProgress: 1, overall: 0.63,
        message: 'ELO CALIBRATED — ENSEMBLE MODEL READY',
        detail: `LR: 35%  NN: 35%  Elo: 20%  EM: 10%  +Player adj`,
      });
      await tick();

      // ── Phase 6: Simulate ─────────────────────────────────────────────
      if (mode === 'single') {
        setProgress({
          phase: 'simulating', phaseProgress: 0, overall: 0.65,
          message: 'SIMULATING TOURNAMENT BRACKET',
          detail: 'Sampling 63 games from learned ensemble probability distribution',
        });
        await tick();

        const simResult = simulateBracket(regionTeams, model, abortRef.current);
        const display   = buildDisplayBracket(regionTeams, model, simResult, null);

        setBracket(display);
        setChampion(simResult.champion);
        setProgress({
          phase: 'done', phaseProgress: 1, overall: 1,
          message: 'BRACKET GENERATED',
          detail: `Predicted champion: ${simResult.champion.name} (#${simResult.champion.seed} seed, ${simResult.champion.region})`,
        });
      } else {
        setProgress({
          phase: 'simulating', phaseProgress: 0, overall: 0.65,
          message: `RUNNING ${simN.toLocaleString()} MONTE CARLO SIMULATIONS`,
          detail: 'Each simulation independently samples all 63 games from ensemble probabilities',
        });
        await tick();

        const consensusData = await runConsensusAnalysis(
          regionTeams, model, simN,
          (done, total) => {
            const frac = done / total;
            setProgress({
              phase: 'simulating',
              phaseProgress: frac,
              overall: 0.65 + frac * 0.28,
              message: `SIMULATING — ${done.toLocaleString()} / ${total.toLocaleString()}`,
              detail: `${(frac * 100).toFixed(1)}% complete`,
            });
          },
          abortRef.current,
        );

        setProgress({
          phase: 'analyzing', phaseProgress: 0, overall: 0.95,
          message: 'AGGREGATING CHAMPION/F4/E8 FREQUENCIES',
          detail: `Computing consensus bracket from ${consensusData.totalSims.toLocaleString()} simulations`,
        });
        await tick();

        const simResult = consensusData.mostLikelyBracket;
        const display   = buildDisplayBracket(regionTeams, model, simResult, consensusData);

        let bestId = ''; let bestCnt = 0;
        consensusData.championFreq.forEach((cnt, id) => { if (cnt > bestCnt) { bestCnt = cnt; bestId = id; } });
        const bestChamp = allTeams.find(t => t.id === bestId) ?? simResult.champion;

        setBracket(display);
        setConsensus(consensusData);
        setChampion(bestChamp);

        setProgress({
          phase: 'done', phaseProgress: 1, overall: 1,
          message: `CONSENSUS COMPLETE — ${consensusData.totalSims.toLocaleString()} SIMULATIONS`,
          detail: `Champion: ${bestChamp.name} — ${((bestCnt / consensusData.totalSims) * 100).toFixed(1)}% of simulations`,
        });
      }
    } catch (err) {
      console.error(err);
      setProgress({
        phase: 'idle', phaseProgress: 0, overall: 0,
        message: 'ERROR DURING SIMULATION',
        detail: String(err),
      });
    } finally {
      setRunning(false);
    }
  }, [running, mode, simN]);

  return (
    <main className="min-h-screen flex flex-col" style={{ background: '#020817', fontFamily: 'monospace' }}>
      {/* ── Header ── */}
      <header style={{ borderBottom: '1px solid #0d1e30', background: '#030b18' }}>
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div style={{ width: 4, height: 32, background: '#f59e0b' }} />
              <div>
                <h1 className="text-2xl font-black tracking-[0.1em] uppercase" style={{ color: '#f0f4f8', letterSpacing: '0.08em' }}>
                  MARCH <span style={{ color: '#f59e0b' }}>MATH</span>NESS
                </h1>
                <p className="text-[10px] tracking-[0.25em] uppercase" style={{ color: '#1e3a5f' }}>
                  Logistic Regression · Neural Network [18→32→16→1] · Elo · Monte Carlo · Player Sim
                </p>
              </div>
            </div>
          </div>

          {modelStats && <ModelBadge stats={modelStats} />}
        </div>
      </header>

      {/* ── Controls ── */}
      <div style={{ borderBottom: '1px solid #0d1e30', background: '#030b18' }}>
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex flex-wrap items-center gap-3">
          {/* Mode buttons */}
          <div style={{ display: 'flex', border: '1px solid #1e3a5f' }}>
            <ModeBtn
              active={mode === 'single'} disabled={running}
              onClick={() => setMode('single')}
              label="GENERATE BRACKET" sub="1 SIMULATION"
            />
            <div style={{ width: 1, background: '#1e3a5f' }} />
            <ModeBtn
              active={mode === 'consensus'} disabled={running}
              onClick={() => setMode('consensus')}
              label="CONSENSUS ANALYSIS" sub={`${simN.toLocaleString()} SIMULATIONS`}
            />
          </div>

          {/* N slider (consensus only) */}
          {mode === 'consensus' && (
            <div
              className="flex items-center gap-3 px-3 py-2"
              style={{ border: '1px solid #1e3a5f', background: '#030b18' }}
            >
              <span className="text-[9px] tracking-[0.2em] text-slate-600 uppercase">SIMS</span>
              <input
                type="range" min={500} max={25000} step={500}
                value={simN} onChange={e => setSimN(+e.target.value)}
                disabled={running}
                className="w-28 accent-amber-500"
                style={{ accentColor: '#f59e0b' }}
              />
              <span className="text-sm font-black tabular-nums" style={{ color: '#f59e0b', minWidth: 48 }}>
                {simN.toLocaleString()}
              </span>
            </div>
          )}

          {/* Generate button */}
          <button
            onClick={run} disabled={running}
            className="px-8 py-2.5 font-black text-xs tracking-[0.2em] uppercase transition-all duration-150"
            style={{
              background: running ? '#0d1e30' : '#f59e0b',
              color: running ? '#1e3a5f' : '#000',
              cursor: running ? 'not-allowed' : 'pointer',
              border: running ? '1px solid #1e3a5f' : '1px solid #f59e0b',
              boxShadow: running ? 'none' : '0 0 16px #f59e0b44',
              letterSpacing: '0.15em',
            }}
          >
            {running ? (
              <span className="flex items-center gap-2">
                <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>◈</span>
                ANALYZING…
              </span>
            ) : mode === 'single' ? 'GENERATE' : `RUN ${simN.toLocaleString()}`}
          </button>

          {/* Data fetch counter */}
          {running && fetchedTeams > 0 && (
            <span className="text-[10px] tracking-widest" style={{ color: '#1e3a5f' }}>
              ESPN: {fetchedTeams}/64 teams loaded
            </span>
          )}
        </div>
      </div>

      {/* ── Progress ── */}
      {progress.phase !== 'idle' && (
        <div style={{ borderBottom: '1px solid #0d1e30', background: '#030b18' }}>
          <div className="max-w-3xl mx-auto px-6 py-3">
            <ProgressBar progress={progress} />
          </div>
        </div>
      )}

      {/* ── Champion banner ── */}
      {champion && !running && (
        <div style={{ borderBottom: '1px solid #0d1e30', background: '#0a0600', padding: '8px 24px' }}>
          <div className="max-w-screen-2xl mx-auto flex items-center gap-4">
            <div style={{ width: 3, height: 28, background: '#f59e0b', flexShrink: 0 }} />
            <span className="text-[9px] tracking-[0.3em] uppercase text-amber-700">
              {mode === 'consensus' ? 'CONSENSUS CHAMPION' : 'PREDICTED CHAMPION'}
            </span>
            <span className="text-base font-black tracking-wide" style={{ color: '#fbbf24' }}>
              #{champion.seed} {champion.name.toUpperCase()}
            </span>
            <span className="text-xs text-amber-800">{champion.region}</span>
            {mode === 'consensus' && consensus && (
              <span className="text-sm font-bold tabular-nums" style={{ color: '#f59e0b' }}>
                {(((consensus.championFreq.get(champion.id) ?? 0) / consensus.totalSims) * 100).toFixed(1)}% OF SIMS
              </span>
            )}
            {champion.roster && champion.roster.length > 0 && (
              <span className="text-[10px] text-slate-700">
                {champion.roster.slice(0, 3).map(p => p.name.split(' ').pop()).join(' · ')}
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Model stats ── */}
      {modelStats && !running && (
        <div style={{ borderBottom: '1px solid #0a1628', background: '#020817' }}>
          <details className="max-w-screen-2xl mx-auto px-6">
            <summary
              className="py-2 text-[9px] tracking-[0.25em] text-slate-700 cursor-pointer uppercase select-none"
              style={{ listStyle: 'none' }}
            >
              ▸ MODEL DETAILS — FEATURE IMPORTANCE · ENSEMBLE WEIGHTS · TRAINING METRICS
            </summary>
            <ModelStatsPanel stats={modelStats} consensus={consensus} allTeams={[]} />
          </details>
        </div>
      )}

      {/* ── Bracket ── */}
      {bracket && !running && (
        <section className="flex-1 px-2 py-4">
          <LegendBar mode={mode} />
          <BracketDisplay bracket={bracket} consensus={consensus} champion={champion} />
        </section>
      )}

      {/* ── Empty state ── */}
      {!bracket && !running && progress.phase === 'idle' && (
        <EmptyState />
      )}

      {/* ── Footer ── */}
      <div style={{ borderTop: '1px solid #0a1628', padding: '6px 24px' }}>
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between">
          <span className="text-[9px] tracking-[0.2em] text-slate-800 uppercase">
            March MathNess · 2025 NCAA Tournament · Player Data via ESPN
          </span>
          <span className="text-[9px] tracking-[0.2em] text-slate-800 uppercase">
            Ensemble: LR + NN + Elo + EM · Player-adjusted probabilities
          </span>
        </div>
      </div>
    </main>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────

function ModeBtn({
  active, disabled, onClick, label, sub,
}: {
  active: boolean; disabled: boolean; onClick: () => void; label: string; sub: string;
}) {
  return (
    <button
      onClick={onClick} disabled={disabled}
      className="px-5 py-2.5 transition-colors duration-100"
      style={{
        background: active ? '#091626' : 'transparent',
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      <div className="text-xs font-bold tracking-[0.1em] uppercase" style={{ color: active ? '#f59e0b' : '#334155' }}>
        {label}
      </div>
      <div className="text-[9px] tracking-widest" style={{ color: active ? '#78716c' : '#1e2d40' }}>
        {sub}
      </div>
    </button>
  );
}

function ModelBadge({ stats }: { stats: ModelStats }) {
  return (
    <div className="flex gap-4">
      {[
        { l: 'LR ACC', v: `${(stats.finalAccuracy * 100).toFixed(1)}%`, c: '#22c55e' },
        { l: 'NN ACC', v: `${((stats.nnAccuracy ?? 0) * 100).toFixed(1)}%`, c: '#3b82f6' },
        { l: 'LR LOSS', v: stats.finalLoss.toFixed(4), c: '#f59e0b' },
        { l: 'SAMPLES', v: stats.trainingSamples.toLocaleString(), c: '#64748b' },
      ].map(m => (
        <div key={m.l} className="text-right">
          <div className="text-[9px] tracking-[0.2em] text-slate-700 uppercase">{m.l}</div>
          <div className="text-sm font-black tabular-nums" style={{ color: m.c }}>{m.v}</div>
        </div>
      ))}
    </div>
  );
}

function ModelStatsPanel({ stats, consensus }: { stats: ModelStats; consensus: ConsensusData | null; allTeams: Team[] }) {
  return (
    <div className="pb-4" style={{ borderTop: '1px solid #0a1628' }}>
      <div className="grid grid-cols-2 gap-4 pt-3">
        {/* Feature importance */}
        <div>
          <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase mb-2">Feature Importance (LR Weights)</div>
          <div className="space-y-1">
            {stats.featureImportance.slice(0, 12).map((fi, i) => (
              <div key={fi.name} className="flex items-center gap-2">
                <span className="text-[8px] text-slate-700 tabular-nums w-4">#{i + 1}</span>
                <div
                  style={{
                    height: 6,
                    width: `${Math.round((fi.absWeight / stats.featureImportance[0].absWeight) * 100)}px`,
                    background: i < 3 ? '#f59e0b' : i < 6 ? '#3b82f6' : '#1e3a5f',
                    flexShrink: 0,
                  }}
                />
                <span className="text-[10px] text-slate-500 flex-1 truncate">{fi.name}</span>
                <span className="text-[9px] tabular-nums" style={{ color: fi.weight >= 0 ? '#22c55e' : '#ef4444' }}>
                  {fi.weight >= 0 ? '+' : ''}{fi.weight.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Consensus stats */}
        <div>
          <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase mb-2">
            {consensus ? 'Consensus Final Four Probabilities' : 'Ensemble Architecture'}
          </div>
          {consensus ? (
            <TopTeamsList freq={consensus.ffFreq} total={consensus.totalSims} label="F4" />
          ) : (
            <EnsembleChart w={stats.ensembleWeights} />
          )}
        </div>
      </div>
    </div>
  );
}

function TopTeamsList({ freq, total, label }: { freq: Map<string, number>; total: number; label: string }) {
  const sorted = Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  return (
    <div className="space-y-0.5">
      {sorted.map(([id, cnt]) => {
        const pct = (cnt / total) * 100;
        return (
          <div key={id} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 capitalize flex-1">{id.replace(/-/g, ' ')}</span>
            <div style={{ width: `${Math.round(pct * 2)}px`, height: 6, background: '#1e3a5f' }} />
            <span className="text-[10px] tabular-nums text-amber-600">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

function EnsembleChart({ w }: { w: { lr: number; nn: number; elo: number; em: number } }) {
  const items = [
    { label: 'Logistic Regression', pct: w.lr * 100, color: '#f59e0b' },
    { label: 'Neural Network', pct: w.nn * 100, color: '#3b82f6' },
    { label: 'Elo System', pct: w.elo * 100, color: '#8b5cf6' },
    { label: 'Efficiency Margin', pct: w.em * 100, color: '#22c55e' },
  ];
  return (
    <div className="space-y-2">
      {items.map(item => (
        <div key={item.label} className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 w-36">{item.label}</span>
          <div style={{ flex: 1, height: 8, background: '#0a1628' }}>
            <div style={{ width: `${item.pct}%`, height: '100%', background: item.color }} />
          </div>
          <span className="text-[10px] tabular-nums font-bold" style={{ color: item.color }}>
            {item.pct.toFixed(0)}%
          </span>
        </div>
      ))}
      <div className="text-[9px] text-slate-700 pt-1">
        +Player adjustment (±8pp) from roster BPM/TS% analysis
      </div>
    </div>
  );
}

function LegendBar({ mode }: { mode: Mode }) {
  return (
    <div className="px-4 mb-2 flex items-center gap-4" style={{ fontFamily: 'monospace' }}>
      <div className="flex items-center gap-1.5">
        <div style={{ width: 3, height: 12, background: '#22c55e' }} />
        <span className="text-[9px] text-slate-600 uppercase tracking-widest">Winner</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div style={{ width: 8, height: 8, background: '#f59e0b' }} />
        <span className="text-[9px] text-slate-600 uppercase tracking-widest">1-seed</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div style={{ width: 8, height: 8, background: '#60a5fa' }} />
        <span className="text-[9px] text-slate-600 uppercase tracking-widest">2–4 seeds</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div style={{ width: 8, height: 8, background: '#a3e635' }} />
        <span className="text-[9px] text-slate-600 uppercase tracking-widest">5–8 seeds</span>
      </div>
      <span className="text-[9px] text-slate-700 uppercase tracking-widest">
        {mode === 'consensus'
          ? '% = consensus simulation frequency'
          : '% = ensemble win probability'}
      </span>
      <span className="text-[9px] text-slate-700 ml-auto">Hover team for player stats</span>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div style={{ border: '1px solid #0d1e30', background: '#030b18', padding: 48, maxWidth: 480 }}>
        <div className="text-[9px] tracking-[0.3em] text-slate-700 uppercase mb-4">System Ready</div>
        <h2 className="text-xl font-black text-slate-300 uppercase tracking-widest mb-4">
          SELECT MODE TO BEGIN
        </h2>
        <div className="space-y-2 text-[11px] text-slate-600">
          <div className="flex gap-3">
            <span style={{ color: '#f59e0b' }}>▸</span>
            <span><strong className="text-slate-400">GENERATE BRACKET</strong> — single ML-powered simulation</span>
          </div>
          <div className="flex gap-3">
            <span style={{ color: '#f59e0b' }}>▸</span>
            <span><strong className="text-slate-400">CONSENSUS ANALYSIS</strong> — up to 25,000 Monte Carlo simulations</span>
          </div>
        </div>
        <div
          className="mt-6 pt-4 space-y-0.5 text-[9px] text-slate-700"
          style={{ borderTop: '1px solid #0d1e30' }}
        >
          {[
            'ESPN live roster data for all 64 tournament teams',
            'Logistic regression: Adam optimizer, 2000 epochs, cosine LR',
            'Neural network: 18→32→16→1, backprop, warm restarts',
            'Elo calibration from NET ranking + efficiency margin',
            'Player-level BPM/TS%/usage adjustments per matchup',
            'Possession simulator for player game-line analysis',
          ].map(s => (
            <div key={s} className="flex gap-2">
              <span style={{ color: '#1e3a5f' }}>–</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function tick() { return new Promise(r => setTimeout(r, 0)); }
