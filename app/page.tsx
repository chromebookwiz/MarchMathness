'use client';

import { useState, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import type { SimProgress, ModelStats, Team } from '@/lib/types';
import type { DisplayBracket, EnsembleModel } from '@/lib/simulation';
import ProgressBar from '@/components/ProgressBar';

const BracketDisplay = dynamic(() => import('@/components/BracketDisplay'), { ssr: false });
const InfoModal      = dynamic(() => import('@/components/InfoModal'),      { ssr: false });
const TeamStatsPanel = dynamic(() => import('@/components/TeamStatsPanel'), { ssr: false });

const DEFAULT_EPOCHS = 1000;

// Minimum ms to show a phase in the progress bar (visual feedback)
async function showPhase(ms: number) {
  await new Promise(r => setTimeout(r, ms));
}

export default function Home() {
  const [nnEpochs, setNnEpochs]       = useState(DEFAULT_EPOCHS);
  const [running, setRunning]         = useState(false);
  const [showInfo, setShowInfo]       = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [progress, setProgress]       = useState<SimProgress>({
    phase: 'idle', phaseProgress: 0, overall: 0,
    message: 'SELECT A MODE AND CLICK GENERATE',
    detail: '',
  });
  const [bracket, setBracket]         = useState<DisplayBracket | null>(null);
  const [champion, setChampion]       = useState<Team | null>(null);
  const [modelStats, setModelStats]   = useState<ModelStats | null>(null);
  const [fetchedTeams, setFetchedTeams] = useState(0);
  const abortRef = useRef({ aborted: false });
  const bracketRef = useRef<HTMLDivElement | null>(null);

  const run = useCallback(async () => {
    if (running) return;
    setRunning(true);
    abortRef.current = { aborted: false };
    setBracket(null); setChampion(null); setSelectedTeam(null);

    try {
      const { buildRegionTeams } = await import('@/lib/bracket');
      const {
        generateTrainingSamples, trainLogisticRegression, computeModelStats,
        simulateBracket, buildDisplayBracket,
        buildEloRatings,
      } = await import('@/lib/simulation');
      const { trainDeepNN } = await import('@/lib/neuralNet');
      const { fetchAllRosters } = await import('@/lib/espn');
      const { fetchMarketOdds } = await import('@/lib/odds');

      const regionTeams = buildRegionTeams();
      const allTeams    = regionTeams.flat();

      // ── Phase 1: Fetch ESPN player data ───────────────────────────────
      setProgress({
        phase: 'fetching', phaseProgress: 0, overall: 0.01,
        message: 'FETCHING LIVE PLAYER DATA FROM ESPN API',
        detail: `Requesting rosters for all ${allTeams.length} tournament teams…`,
      });
      await showPhase(200);

      const rosterMap = await fetchAllRosters(
        allTeams.map(t => t.id),
        (done, total) => {
          setFetchedTeams(done);
          setProgress({
            phase: 'fetching',
            phaseProgress: done / total,
            overall: 0.01 + (done / total) * 0.13,
            message: `FETCHING ROSTERS — ${done}/${total} TEAMS`,
            detail: `${allTeams[done - 1]?.name ?? ''} loaded`,
          });
        },
      );

      for (const team of allTeams) {
        team.roster = rosterMap.get(team.id) ?? [];
      }

      const withRosters = allTeams.filter(t => (t.roster?.length ?? 0) > 0).length;
      const totalPlayers = allTeams.reduce((s, t) => s + (t.roster?.length ?? 0), 0);

      setProgress({
        phase: 'fetching', phaseProgress: 1, overall: 0.14,
        message: `PLAYER DATA LOADED — ${withRosters}/${allTeams.length} TEAMS`,
        detail: `${totalPlayers} players • BPM / TS% / usage extracted • Player adjustments active`,
      });
      await showPhase(300);

      // ── Phase 2: Generate training samples ────────────────────────────
      setProgress({
        phase: 'generating', phaseProgress: 0, overall: 0.15,
        message: 'GENERATING SYNTHETIC SEASON GAME LOGS',
        detail: 'Q1-Q4 opponent distribution • H2H peer matchups • player-adjusted labels',
      });
      await showPhase(150);

      const samples   = generateTrainingSamples(allTeams);
      const samplesNN = [...samples];

      setProgress({
        phase: 'generating', phaseProgress: 0.5, overall: 0.17,
        message: 'BUILDING TRAINING CORPUS',
        detail: `${samples.length.toLocaleString()} labelled matchup samples generated`,
      });
      await showPhase(150);

      setProgress({
        phase: 'generating', phaseProgress: 1, overall: 0.20,
        message: `SEASON DATA READY — ${samples.length.toLocaleString()} SAMPLES`,
        detail: 'Feature vectors: 18-dimensional • Label smoothing: ε=0.05',
      });
      await showPhase(200);

      // ── Phase 3: Train Logistic Regression ────────────────────────────
      setProgress({
        phase: 'training-lr', phaseProgress: 0, overall: 0.21,
        message: 'TRAINING LOGISTIC REGRESSION — ADAM OPTIMIZER',
        detail: 'β₁=0.9  β₂=0.999  λ=0.0008  cosine LR decay  2000 epochs  batch=64',
      });
      await showPhase(100);

      let lrFinalLoss = 1, lrFinalAcc = 0;
      const lrWeights = await trainLogisticRegression(samples, tp => {
        lrFinalLoss = tp.loss; lrFinalAcc = tp.accuracy;
        setProgress({
          phase: 'training-lr',
          phaseProgress: tp.epoch / tp.totalEpochs,
          overall: 0.21 + (tp.epoch / tp.totalEpochs) * 0.20,
          message: `LR GRADIENT DESCENT — EPOCH ${tp.epoch.toLocaleString()}/${tp.totalEpochs.toLocaleString()}`,
          detail: `Loss: ${tp.loss.toFixed(5)}  Acc: ${(tp.accuracy * 100).toFixed(1)}%  LR: ${tp.lrDecay.toFixed(4)}`,
          training: tp,
        });
      });

      // ── Phase 4: Train Deep Neural Network ───────────────────────────
      setProgress({
        phase: 'training-nn', phaseProgress: 0, overall: 0.42,
        message: 'TRAINING DEEP NEURAL NETWORK [18→64→32→16→8→1]',
        detail: 'LeakyReLU α=0.01 · label smoothing ε=0.05 · gradient clipping · cosine warm restarts · 1000 epochs',
      });
      await showPhase(100);

      let nnFinalAcc = 0;
      const nnWeights = await trainDeepNN(samplesNN, (epoch, totalEpochs, loss, acc, lr) => {
        nnFinalAcc = acc;
        setProgress({
          phase: 'training-nn',
          phaseProgress: epoch / totalEpochs,
          overall: 0.42 + (epoch / totalEpochs) * 0.18,
          message: `DEEP NN BACKPROP — EPOCH ${epoch.toLocaleString()}/${totalEpochs.toLocaleString()}`,
          detail: `Loss: ${loss.toFixed(5)}  Acc: ${(acc * 100).toFixed(1)}%  LR: ${lr.toFixed(5)}`,
          training: { epoch, totalEpochs, loss, accuracy: acc, lrDecay: lr, modelType: 'neural' },
        });
      }, nnEpochs);

      // ── Phase 5: Calibrate Elo ────────────────────────────────────────
      setProgress({
        phase: 'calibrating-elo', phaseProgress: 0, overall: 0.61,
        message: 'CALIBRATING ELO RATINGS FROM SEASON PERFORMANCE',
        detail: 'NET rank × 6pt/step + efficiency margin × 12 + SOS bonus → Elo base 1500',
      });
      await showPhase(250);

      const elos  = buildEloRatings(allTeams);

      setProgress({
        phase: 'calibrating-elo', phaseProgress: 0.2, overall: 0.64,
        message: 'FETCHING MARKET ODDS',
        detail: 'Retrieving live betting prices from odds providers',
      });
      await showPhase(150);

      const marketOdds = await fetchMarketOdds(allTeams);

      const stats = computeModelStats(lrWeights, samples, lrFinalLoss, lrFinalAcc, nnFinalAcc);
      setModelStats(stats);

      const model: EnsembleModel = {
        lrWeights,
        nnWeights,
        elos,
        marketOdds,
        ensembleW: { lr: 0.35, nn: 0.35, elo: 0.20, em: 0.10, market: 0.10 },
      };

      setProgress({
        phase: 'calibrating-elo', phaseProgress: 0.5, overall: 0.62,
        message: 'BUILDING ENSEMBLE MODEL',
        detail: 'LR 35% · Deep NN 35% · Elo 20% · Efficiency Margin 10% · ±8pp Player Adj',
      });
      await showPhase(300);

      setProgress({
        phase: 'calibrating-elo', phaseProgress: 1, overall: 0.64,
        message: 'ENSEMBLE MODEL READY',
        detail: `LR acc: ${(lrFinalAcc * 100).toFixed(1)}%  NN acc: ${(nnFinalAcc * 100).toFixed(1)}%  Elo: calibrated`,
      });
      await showPhase(200);

        setProgress({
          phase: 'simulating', phaseProgress: 0, overall: 0.66,
          message: 'SIMULATING TOURNAMENT BRACKET',
          detail: 'Evaluating 63 games deterministically from ensemble probability distribution',
        });
        await showPhase(300);

        const simResult = simulateBracket(regionTeams, model, abortRef.current);

        setProgress({
          phase: 'simulating', phaseProgress: 0.6, overall: 0.80,
          message: 'RUNNING PLAYER-LEVEL POSSESSION SIMULATION',
          detail: 'BPM · TS% · depth score · usage-weighted shot selection',
        });
        await showPhase(300);

        setProgress({
          phase: 'simulating', phaseProgress: 1, overall: 0.90,
          message: 'BRACKET SIMULATION COMPLETE',
          detail: `63 games resolved · Champion: ${simResult.champion.name}`,
        });
        await showPhase(150);

        setProgress({
          phase: 'analyzing', phaseProgress: 0, overall: 0.92,
          message: 'BUILDING DISPLAY BRACKET',
          detail: 'Mapping game results to visualization layout',
        });
        await showPhase(200);

        const display = buildDisplayBracket(regionTeams, model, simResult);
        setBracket(display);
        setChampion(simResult.champion);

        setProgress({
          phase: 'done', phaseProgress: 1, overall: 1,
          message: 'BRACKET GENERATED',
          detail: `Predicted champion: ${simResult.champion.name} (#${simResult.champion.seed} seed, ${simResult.champion.region})`,
        });

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
  }, [running, nnEpochs]);

  const downloadPdf = useCallback(async () => {
    if (!bracketRef.current) return;
    const { downloadElementPdf } = await import('@/lib/pdf');
    await downloadElementPdf(bracketRef.current, 'march-mathness-bracket');
  }, []);

  return (
    <main className="min-h-screen flex flex-col">

      {/* ── Header ── */}
      <header style={{ borderBottom: '1px solid #0d1e30', background: '#030b18' }}>
        <div className="flex items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold">March Mathness Bracket Generator</h1>
          </div>

          <div className="flex items-center gap-4">
            {modelStats && <ModelBadge stats={modelStats} />}
            <button
              onClick={() => setShowInfo(true)}
              style={{
                border: '1px solid #1e3a5f',
                background: 'transparent',
                color: '#475569',
                width: 28,
                height: 28,
                fontFamily: 'monospace',
                fontSize: 13,
                fontWeight: 700,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
              title="About this simulation"
            >
              ?
            </button>
          </div>
        </div>
      </header>

      {/* ── Controls ── */}
      <div style={{ borderBottom: '1px solid #0d1e30', background: '#030b18' }}>
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex flex-wrap items-center gap-3">
          {/* Neural Net Epochs Input */}
          <div
            className="flex items-center px-3 py-1.5"
            style={{ border: '1px solid #1e3a5f', background: '#030b18' }}
          >
            <span className="text-[10px] tracking-[0.1em] text-slate-400 uppercase mr-3">NN Epochs</span>
            <input
              type="number"
              min={1}
              max={20000}
              value={nnEpochs}
              onChange={e => {
                let v = parseInt(e.target.value, 10);
                if (isNaN(v)) v = 1000;
                v = Math.max(1, Math.min(20000, v));
                setNnEpochs(v);
              }}
              disabled={running}
              style={{
                width: '74px',
                background: 'transparent',
                border: '1px solid #1e3a5f',
                color: '#f59e0b',
                fontFamily: 'monospace',
                fontSize: 13,
                fontWeight: 900,
                padding: '4px 8px',
                textAlign: 'center',
              }}
            />
          </div>

          {/* Generate button */}
          <button
            onClick={run} disabled={running}
            style={{
              padding: '10px 32px',
              fontFamily: 'monospace',
              fontWeight: 900,
              fontSize: 12,
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              background: running ? '#0d1e30' : '#f59e0b',
              color: running ? '#1e3a5f' : '#000',
              cursor: running ? 'not-allowed' : 'pointer',
              border: running ? '1px solid #1e3a5f' : '1px solid #f59e0b',
              boxShadow: running ? 'none' : '0 0 16px #f59e0b44',
              transition: 'all 0.15s',
            }}
          >
            {running ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ animation: 'spin 1s linear infinite', display: 'inline-block' }}>◈</span>
                ANALYZING…
              </span>
            ) : 'GENERATE BRACKET'}
          </button>

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
              PREDICTED CHAMPION
            </span>
            <span className="text-base font-black tracking-wide" style={{ color: '#fbbf24' }}>
              #{champion.seed} {champion.name.toUpperCase()}
            </span>
            <span className="text-xs text-amber-800">{champion.region}</span>
            {champion.roster && champion.roster.length > 0 && (
              <span className="text-[10px] text-slate-700">
                {champion.roster.slice(0, 3).map(p => p.name.split(' ').pop()).join(' · ')}
              </span>
            )}
            <button
              onClick={() => setSelectedTeam(champion)}
              style={{
                marginLeft: 'auto',
                border: '1px solid #f59e0b44',
                background: 'transparent',
                color: '#f59e0b',
                fontSize: 9,
                letterSpacing: '0.15em',
                padding: '3px 10px',
                cursor: 'pointer',
                fontFamily: 'monospace',
              }}
            >
              VIEW STATS
            </button>
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
            <ModelStatsPanel stats={modelStats} />
          </details>
        </div>
      )}

      {/* ── Bracket ── */}
      {bracket && !running && (
        <section ref={bracketRef} className="flex-1 px-2 py-4">
          <LegendBar />
          <div className="flex items-center gap-3 mb-3">
            <button
              onClick={downloadPdf}
              style={{
                padding: '8px 18px',
                fontFamily: 'monospace',
                fontWeight: 700,
                fontSize: 12,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                background: '#0f172a',
                color: '#f59e0b',
                border: '1px solid #1e3a5f',
                cursor: 'pointer',
              }}
            >
              Download PDF
            </button>
            <span className="text-sm text-slate-500">(saves current bracket view)</span>
          </div>
          <BracketDisplay
            bracket={bracket}
            champion={champion}
            onTeamClick={setSelectedTeam}
          />
        </section>
      )}

      {/* ── Empty state ── */}
      {!bracket && !running && progress.phase === 'idle' && (
        <EmptyState onInfo={() => setShowInfo(true)} />
      )}

      {/* ── Footer ── */}
      <div style={{ borderTop: '1px solid #0a1628', padding: '6px 24px' }}>
        <div className="flex items-center justify-between">
          <span>March Mathness · 2026 NCAA Tournament</span>
          <span>Player Data via ESPN</span>
        </div>
      </div>

      {/* ── Modals & Panels ── */}
      <InfoModal isOpen={showInfo} onClose={() => setShowInfo(false)} />
      <TeamStatsPanel
        team={selectedTeam}
        onClose={() => setSelectedTeam(null)}
      />
    </main>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function ModeBtn({
  active, disabled, onClick, label, sub,
}: {
  active: boolean; disabled: boolean; onClick: () => void; label: string; sub: string;
}) {
  return (
    <button
      onClick={onClick} disabled={disabled}
      style={{
        padding: '10px 20px',
        background: active ? '#091626' : 'transparent',
        cursor: disabled ? 'not-allowed' : 'pointer',
        border: 'none',
        fontFamily: 'monospace',
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: active ? '#f59e0b' : '#334155' }}>
        {label}
      </div>
      <div style={{ fontSize: 9, letterSpacing: '0.15em', color: active ? '#78716c' : '#1e2d40' }}>
        {sub}
      </div>
    </button>
  );
}

function ModelBadge({ stats }: { stats: ModelStats }) {
  return (
    <div className="flex gap-4">
      {[
        { l: 'LR ACC',   v: `${(stats.finalAccuracy * 100).toFixed(1)}%`,          c: '#22c55e' },
        { l: 'NN ACC',   v: `${((stats.nnAccuracy ?? 0) * 100).toFixed(1)}%`,      c: '#3b82f6' },
        { l: 'LR LOSS',  v: stats.finalLoss.toFixed(4),                             c: '#f59e0b' },
        { l: 'SAMPLES',  v: stats.trainingSamples.toLocaleString(),                 c: '#64748b' },
      ].map(m => (
        <div key={m.l} className="text-right">
          <div className="text-[9px] tracking-[0.2em] text-slate-700 uppercase">{m.l}</div>
          <div className="text-sm font-black tabular-nums" style={{ color: m.c }}>{m.v}</div>
        </div>
      ))}
    </div>
  );
}

function ModelStatsPanel({ stats }: { stats: ModelStats }) {
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

        {/* Ensemble */}
        <div>
          <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase mb-2">
            Ensemble Architecture
          </div>
          <EnsembleChart w={stats.ensembleWeights} />
        </div>
      </div>
    </div>
  );
}

function TopTeamsList({ freq, total }: { freq: Map<string, number>; total: number }) {
  const sorted = Array.from(freq.entries()).sort((a, b) => b[1] - a[1]).slice(0, 8);
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
    { label: 'Deep Neural Network', pct: w.nn * 100, color: '#3b82f6' },
    { label: 'Elo System',          pct: w.elo * 100, color: '#8b5cf6' },
    { label: 'Efficiency Margin',   pct: w.em * 100, color: '#22c55e' },
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
        +Player adjustment (±8pp) via BPM / TS% / depth score
      </div>
    </div>
  );
}

function LegendBar() {
  return (
    <div className="px-4 mb-2 flex items-center gap-4" style={{ fontFamily: 'monospace' }}>
      {[
        { color: '#22c55e', label: 'Winner' },
        { color: '#f59e0b', label: '1-seed' },
        { color: '#60a5fa', label: '2–4 seeds' },
        { color: '#a3e635', label: '5–8 seeds' },
      ].map(({ color, label }) => (
        <div key={label} className="flex items-center gap-1.5">
          <div style={{ width: 8, height: 8, background: color }} />
          <span className="text-[9px] text-slate-600 uppercase tracking-widest">{label}</span>
        </div>
      ))}
      <span className="text-[9px] text-slate-700 uppercase tracking-widest">
        % = ensemble win probability
      </span>
      <span className="text-[9px] text-slate-700 ml-auto">Click team for full stats · Hover for quick view</span>
    </div>
  );
}

function EmptyState({ onInfo }: { onInfo: () => void }) {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div style={{ border: '1px solid #0d1e30', background: '#030b18', padding: 48, maxWidth: 520 }}>
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
            'Logistic Regression: Adam · 2000 epochs · cosine LR · λ=0.0008',
            'Deep Neural Network: [18→64→32→16→8→1] · LeakyReLU · 1000 epochs',
            'Elo calibration from NET ranking + efficiency margin + SOS',
            'Player-level BPM / TS% / usage adjustments per matchup',
            'Possession-by-possession player simulation with OT support',
          ].map(s => (
            <div key={s} className="flex gap-2">
              <span style={{ color: '#1e3a5f' }}>–</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
        <button
          onClick={onInfo}
          style={{
            marginTop: 24,
            border: '1px solid #1e3a5f',
            background: 'transparent',
            color: '#475569',
            fontSize: 9,
            letterSpacing: '0.2em',
            padding: '6px 16px',
            cursor: 'pointer',
            fontFamily: 'monospace',
          }}
        >
          LEARN HOW IT WORKS →
        </button>
      </div>
    </div>
  );
}
