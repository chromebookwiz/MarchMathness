'use client';

import { useState } from 'react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

type Tab = 'overview' | 'data' | 'models' | 'simulation' | 'statistics';

const TABS: { id: Tab; label: string }[] = [
  { id: 'overview',   label: 'OVERVIEW'   },
  { id: 'data',       label: 'DATA'       },
  { id: 'models',     label: 'ML MODELS'  },
  { id: 'simulation', label: 'SIMULATION' },
  { id: 'statistics', label: 'STATISTICS' },
];

export default function InfoModal({ isOpen, onClose }: Props) {
  const [tab, setTab] = useState<Tab>('overview');

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 99999,
        background: 'rgba(2,8,23,0.88)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: 'monospace',
      }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          width: 700, maxHeight: '88vh',
          border: '1px solid #1e3a5f',
          background: '#040d1a',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px',
          borderBottom: '1px solid #1e3a5f',
          background: '#030b18',
        }}>
          <div>
            <div style={{ fontSize: 9, color: '#1e3a5f', letterSpacing: '0.3em', textTransform: 'uppercase' }}>
              Documentation
            </div>
            <div style={{ fontSize: 14, fontWeight: 900, color: '#f0f4f8', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              March <span style={{ color: '#f59e0b' }}>Math</span>Ness — Simulation Engine
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent', border: '1px solid #1e3a5f',
              color: '#475569', width: 28, height: 28,
              cursor: 'pointer', fontSize: 14, fontFamily: 'monospace',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            ×
          </button>
        </div>

        {/* Tab bar */}
        <div style={{ display: 'flex', borderBottom: '1px solid #1e3a5f', background: '#030b18' }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: '8px 16px',
                background: 'transparent', border: 'none',
                borderBottom: tab === t.id ? '2px solid #f59e0b' : '2px solid transparent',
                color: tab === t.id ? '#f59e0b' : '#334155',
                fontSize: 10, fontWeight: 700, letterSpacing: '0.15em',
                cursor: 'pointer', fontFamily: 'monospace',
                marginBottom: -1,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ overflow: 'auto', padding: 24, flex: 1 }}>
          {tab === 'overview'   && <OverviewTab />}
          {tab === 'data'       && <DataTab />}
          {tab === 'models'     && <ModelsTab />}
          {tab === 'simulation' && <SimulationTab />}
          {tab === 'statistics' && <StatisticsTab />}
        </div>
      </div>
    </div>
  );
}

// ── Shared primitives ──────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        fontSize: 9, color: '#f59e0b', letterSpacing: '0.25em',
        textTransform: 'uppercase', marginBottom: 8,
        borderBottom: '1px solid #0d1e30', paddingBottom: 4,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontSize: 11, color: '#64748b', lineHeight: 1.7, marginBottom: 8 }}>
      {children}
    </p>
  );
}

function Bullet({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: 10, marginBottom: 6 }}>
      <span style={{ color: '#f59e0b', fontSize: 10, flexShrink: 0 }}>▸</span>
      <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>
        <strong style={{ color: '#94a3b8' }}>{label}</strong>{children}
      </div>
    </div>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code style={{
      background: '#08111e', border: '1px solid #0d1e30',
      padding: '1px 5px', fontSize: 10, color: '#7dd3fc',
    }}>
      {children}
    </code>
  );
}

function StatRow({ label, value, color = '#94a3b8' }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', borderBottom: '1px solid #0a1628' }}>
      <span style={{ fontSize: 10, color: '#475569' }}>{label}</span>
      <span style={{ fontSize: 10, color, fontWeight: 700 }}>{value}</span>
    </div>
  );
}

// ── Tab content ────────────────────────────────────────────────────────────

function OverviewTab() {
  return (
    <div>
      <Section title="What this is">
        <P>
          March MathNess is a statistically rigorous NCAA tournament bracket prediction engine that combines
          live player data, machine learning, Elo ratings, and Monte Carlo simulation to produce the most
          analytically grounded bracket predictions available.
        </P>
        <P>
          Rather than relying on any single model, it uses an <strong style={{ color: '#e2e8f0' }}>ensemble</strong> of
          four distinct systems — each capturing a different aspect of team quality — then applies player-level
          adjustments based on individual BPM, true shooting, and usage data.
        </P>
      </Section>

      <Section title="How it works — 7 phases">
        {[
          ['1. FETCH', 'Pull live roster data for all 64 tournament teams from the ESPN API. Each player is parsed for PPG, RPG, APG, BPM, TS%, usage rate, and shooting splits.'],
          ['2. GENERATE', 'Synthesize ~2,400 labelled game samples across Q1–Q4 opponent distributions. Each team plays H2H matchups vs peers and quality-band opponents.'],
          ['3. TRAIN LR', 'Fit a logistic regression model using Adam optimizer (2000 epochs). The 18 feature weights capture efficiency margin, quadrant records, four factors, and more.'],
          ['4. TRAIN NN', 'Train a 5-layer deep neural network [18→64→32→16→8→1] with LeakyReLU, label smoothing, gradient clipping, and cosine annealing with warm restarts.'],
          ['5. CALIBRATE ELO', 'Initialize Elo ratings from NET ranking, efficiency margin, and strength of schedule. Used as a third independent predictor.'],
          ['6. SIMULATE', 'Run up to 25,000 independent bracket simulations. Each game is resolved by sampling from the ensemble probability distribution.'],
          ['7. AGGREGATE', 'Compute champion/F4/E8/S16 frequencies, Wilson confidence intervals, expected wins, and build the consensus most-likely bracket.'],
        ].map(([label, desc]) => (
          <Bullet key={label} label={`${label} — `}>{desc}</Bullet>
        ))}
      </Section>

      <Section title="Key capabilities">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {[
            ['18 features', 'per matchup'],
            ['1,000 NN epochs', 'per run'],
            ['25,000 max sims', 'Monte Carlo'],
            ['64 teams', 'live ESPN data'],
            ['63 games', 'per simulation'],
            ['Player BPM adj', '±8pp per game'],
          ].map(([val, label]) => (
            <div key={val} style={{ background: '#08111e', border: '1px solid #0d1e30', padding: '8px 12px' }}>
              <div style={{ fontSize: 16, fontWeight: 900, color: '#f59e0b' }}>{val}</div>
              <div style={{ fontSize: 9, color: '#475569', letterSpacing: '0.15em', textTransform: 'uppercase' }}>{label}</div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

function DataTab() {
  return (
    <div>
      <Section title="ESPN API — live player data">
        <P>
          All 64 tournament team rosters are fetched from the ESPN unofficial API via a server-side proxy
          (avoiding CORS). Data is cached in localStorage for 6 hours to prevent redundant requests.
          If ESPN returns empty data, a deterministic fallback roster is generated from the team ID seed.
        </P>
        <P>
          <strong style={{ color: '#e2e8f0' }}>Per-player stats extracted:</strong>
        </P>
        {[
          ['PPG / RPG / APG / SPG / BPG / TOV', 'Standard per-game box score stats'],
          ['FG% / 3P% / FT%', 'Shooting efficiency splits'],
          ['True Shooting %', 'TS% = PTS / (2 × (FGA + 0.44 × FTA))'],
          ['Usage Rate', '% of team possessions used while on court'],
          ['BPM (approx)', 'Box Plus/Minus estimated from per-game splits'],
          ['OBPM / DBPM', 'Offensive and defensive BPM components'],
          ['Star Score', 'Composite 0–100 rating combining PPG, BPM, TS%'],
        ].map(([label, desc]) => <Bullet key={label} label={label + ': '}>{desc}</Bullet>)}
      </Section>

      <Section title="Pre-loaded team metrics">
        <P>
          KenPom-style adjusted efficiency metrics are pre-loaded for all 64 teams (2026 NCAA field).
          These metrics are the foundation of the feature vectors used in all ML models.
        </P>
        {[
          ['AdjOE', 'Adjusted offensive efficiency (points per 100 possessions)'],
          ['AdjDE', 'Adjusted defensive efficiency — lower is better'],
          ['AdjTempo', 'Adjusted pace (possessions per 40 minutes)'],
          ['NET Ranking', 'NCAA Evaluation Tool ranking (1–64)'],
          ['Quadrant Records', 'Q1/Q2/Q3/Q4 wins & losses by opponent quality'],
          ['Strength of Schedule', 'Composite SOS rank'],
          ['Four Factors', 'EFG%, TO rate, ORB%, FT rate (offense + defense)'],
          ['Experience / Coach', 'Team experience score + coach tournament win history'],
        ].map(([label, desc]) => <Bullet key={label} label={label + ': '}>{desc}</Bullet>)}
      </Section>
    </div>
  );
}

function ModelsTab() {
  return (
    <div>
      <Section title="Logistic Regression — 35% ensemble weight">
        <P>
          A linear model trained on 18 handcrafted features. Despite its simplicity, logistic regression
          is extremely competitive for sports prediction because the features themselves encode domain expertise.
        </P>
        <div style={{ background: '#08111e', border: '1px solid #0d1e30' }}>
          <StatRow label="Optimizer" value="Adam (β₁=0.9, β₂=0.999, ε=1e-8)" />
          <StatRow label="Epochs" value="2,000" />
          <StatRow label="Batch size" value="64" />
          <StatRow label="Regularization" value="L2 λ=0.0008" />
          <StatRow label="LR schedule" value="Cosine annealing (0.015 → 0)" />
          <StatRow label="Features" value="18 (efficiency, quadrant, four factors, etc.)" />
        </div>
      </Section>

      <Section title="Deep Neural Network — 35% ensemble weight">
        <P>
          A 5-layer feedforward network that can learn non-linear feature interactions. The architecture
          is designed to extract hierarchical representations of matchup quality.
        </P>
        <div style={{ background: '#08111e', border: '1px solid #0d1e30', marginBottom: 8 }}>
          <StatRow label="Architecture" value="[18 → 64 → 32 → 16 → 8 → 1]" color="#3b82f6" />
          <StatRow label="Hidden activation" value="LeakyReLU (α=0.01)" />
          <StatRow label="Output activation" value="Sigmoid" />
          <StatRow label="Weight storage" value="Float32Array (2× faster)" />
          <StatRow label="Label smoothing" value="ε=0.05 (prevents overconfidence)" />
          <StatRow label="Gradient clipping" value="L2 norm ≤ 1.0 (training stability)" />
          <StatRow label="LR schedule" value="Linear warmup (40 epochs) + cosine × 3 restarts" />
          <StatRow label="Epochs" value="1,000" />
          <StatRow label="Batch size" value="64" />
          <StatRow label="Optimizer" value="Adam per-layer (β₁=0.9, β₂=0.999)" />
        </div>
        <P>
          He initialization (<Mono>σ = √(2/fan_in)</Mono>) via Box-Muller transform ensures
          proper gradient flow from the start.
        </P>
      </Section>

      <Section title="Elo Rating System — 20% ensemble weight">
        <P>
          Each team is assigned an Elo rating based on their season performance.
          The win probability follows the standard Elo formula.
        </P>
        <P>
          <Mono>Elo = 1500 + NET_bonus + EM_bonus + SOS_bonus</Mono>
        </P>
        <P>
          <Mono>P(A beats B) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))</Mono>
        </P>
      </Section>

      <Section title="Efficiency Margin — 10% ensemble weight">
        <P>
          Raw adjusted efficiency differential (<Mono>AdjOE − AdjDE</Mono>) passed through sigmoid.
          A simple but historically predictive baseline: larger efficiency margins strongly correlate with tournament success.
        </P>
      </Section>

      <Section title="Player adjustment — ±8pp">
        <P>
          After ensemble blending, each game's probability is adjusted by up to ±8 percentage points
          based on roster-level analysis: star player BPM differential, depth score, true shooting
          differential, and defensive BPM. This captures player-quality effects not visible in team aggregates.
        </P>
      </Section>
    </div>
  );
}

function SimulationTab() {
  return (
    <div>
      <Section title="Monte Carlo bracket simulation">
        <P>
          In Consensus mode, the engine runs N independent full-bracket simulations. Each simulation
          independently resolves all 63 games by sampling from ensemble probabilities — meaning an
          upset can (and does) occur in any game, reflecting true tournament chaos.
        </P>
        <P>
          After N simulations, champion/F4/E8/S16 frequencies are computed from the aggregate.
          The most-likely consensus bracket is built using a sharpened model (6× LR weights) to
          deterministically pick the highest-probability winner at each slot.
        </P>
        <div style={{ background: '#08111e', border: '1px solid #0d1e30' }}>
          <StatRow label="Max simulations" value="25,000" />
          <StatRow label="Batch size" value="150 sims / tick (non-blocking)" />
          <StatRow label="Games per sim" value="63" />
          <StatRow label="Total games (25k)" value="1,575,000" color="#f59e0b" />
          <StatRow label="Probability floor" value="3% / 97% (no coinflips)" />
        </div>
      </Section>

      <Section title="Possession-level player simulation">
        <P>
          Each game can also be simulated at the possession level. The simulator models individual
          player shot selection, defensive adjustments, blocks, offensive rebounds, fouls, and free throws.
        </P>
        {[
          ['Possessions', 'avgTempo × 0.94 possessions per team per game'],
          ['Shot selection', 'Usage-weighted random player selection'],
          ['3PT vs 2PT', 'Player tendency + defensive 3PT rate adjustment'],
          ['Defensive adj', 'leagueAvgEFG / (defTeam.defEfgPct/100)'],
          ['Blocks', 'Defender BPG probability applied to each shot'],
          ['ORB put-backs', 'Offensive rebound rate determines second chances'],
          ['Fouls / FTs', 'FTA rate triggers free throw sequences'],
          ['Overtime', 'Full OT simulated if regulation ends tied'],
        ].map(([label, desc]) => <Bullet key={label} label={label + ': '}>{desc}</Bullet>)}
      </Section>
    </div>
  );
}

function StatisticsTab() {
  return (
    <div>
      <Section title="Brier Score">
        <P>
          The Brier Score measures the mean squared error between predicted probabilities and outcomes.
          A perfect model scores 0.0; a random model scores 0.25. Lower is better.
        </P>
        <P>
          <Mono>BS = (1/N) × Σ (p_i − o_i)²</Mono>
        </P>
      </Section>

      <Section title="Expected Calibration Error (ECE)">
        <P>
          ECE measures how well predicted probabilities match observed frequencies. If you predict
          70% win probability for 100 games, a perfectly calibrated model would see ~70 wins.
          ECE is the weighted average miscalibration across probability bins.
        </P>
        <P>
          <Mono>ECE = Σ_b (|b| / N) × |avg_prob(b) − avg_outcome(b)|</Mono>
        </P>
      </Section>

      <Section title="Wilson Confidence Intervals">
        <P>
          Rather than simple proportion estimates, Wilson confidence intervals account for small sample
          sizes and boundary effects. A team predicted to win 8% of simulations out of 5,000 has a
          narrower CI than one with 2% out of 500.
        </P>
        <P>
          95% CIs are shown as <Mono>[lower, upper]</Mono> bounds on champion probability.
        </P>
      </Section>

      <Section title="Expected Wins">
        <P>
          Expected wins approximates how many games a team will win in expectation across all possible
          bracket outcomes:
        </P>
        <P>
          <Mono>E[wins] = champ×6 + (F4−champ)×5 + (E8−F4)×4 + (S16−E8)×3</Mono>
        </P>
        <P>
          A #1 seed with 25% champion probability, 55% F4 rate, 75% E8, 90% S16 would have
          approximately 4.2 expected wins.
        </P>
      </Section>

      <Section title="Interpreting probabilities">
        <P>
          <strong style={{ color: '#e2e8f0' }}>Single bracket mode:</strong> percentages shown are the ensemble&apos;s raw
          win probability for each game (team A vs team B). These are not frequencies but model outputs.
        </P>
        <P>
          <strong style={{ color: '#e2e8f0' }}>Consensus mode:</strong> percentages reflect empirical frequencies
          from N simulations. A team showing 23% is the consensus winner of that slot in 23% of all
          bracket simulations run.
        </P>
      </Section>
    </div>
  );
}
