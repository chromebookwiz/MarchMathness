'use client';

import { SLOT_H, ROUND_W, REGION_H, gameTopPx, seedColor, winProbColor } from '@/lib/bracket';
import type { DisplayBracket, DisplayGame } from '@/lib/simulation';
import type { Team, Player } from '@/lib/types';
import type { ConsensusData } from '@/lib/simulation';
import { useState } from 'react';

interface Props {
  bracket: DisplayBracket;
  consensus: ConsensusData | null;
  champion: Team | null;
  onTeamClick?: (team: Team) => void;
}

export default function BracketDisplay({ bracket, consensus, champion, onTeamClick }: Props) {
  const [hoveredTeam, setHoveredTeam] = useState<Team | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  return (
    <div className="w-full overflow-x-auto pb-6 relative" style={{ fontFamily: 'monospace' }}>
      {/* Scanline overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-50 opacity-[0.03]"
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.8) 2px, rgba(255,255,255,0.8) 3px)',
        }}
      />

      <div
        className="inline-flex items-start gap-0 min-w-max mx-auto"
        style={{ paddingTop: '2.5rem', paddingBottom: '2.5rem' }}
        onMouseLeave={() => setHoveredTeam(null)}
      >
        {/* LEFT: East (top) + South (bottom) */}
        <div className="flex flex-col" style={{ gap: 20 }}>
          <RegionLabel name="EAST" />
          <RegionHalf rounds={bracket.east} direction="ltr" onHover={setHoveredTeam} setPos={setTooltipPos} onClick={onTeamClick} />
          <RegionLabel name="SOUTH" />
          <RegionHalf rounds={bracket.south} direction="ltr" onHover={setHoveredTeam} setPos={setTooltipPos} onClick={onTeamClick} />
        </div>

        {/* CENTER */}
        <CenterColumn
          bracket={bracket}
          consensus={consensus}
          champion={champion}
          onHover={setHoveredTeam}
          setPos={setTooltipPos}
          onClick={onTeamClick}
        />

        {/* RIGHT: Midwest (top) + West (bottom) */}
        <div className="flex flex-col" style={{ gap: 20 }}>
          <RegionLabel name="MIDWEST" right />
          <RegionHalf rounds={bracket.midwest} direction="rtl" onHover={setHoveredTeam} setPos={setTooltipPos} onClick={onTeamClick} />
          <RegionLabel name="WEST" right />
          <RegionHalf rounds={bracket.west} direction="rtl" onHover={setHoveredTeam} setPos={setTooltipPos} onClick={onTeamClick} />
        </div>
      </div>

      {/* Player tooltip */}
      {hoveredTeam && (
        <PlayerTooltip team={hoveredTeam} x={tooltipPos.x} y={tooltipPos.y} />
      )}
    </div>
  );
}

function RegionLabel({ name, right = false }: { name: string; right?: boolean }) {
  return (
    <div
      className="text-[9px] font-bold tracking-[0.3em] text-slate-600 border-b border-[#0d1e30] pb-1"
      style={{ textAlign: right ? 'right' : 'left' }}
    >
      {name}
    </div>
  );
}

function RegionHalf({
  rounds, direction, onHover, setPos, onClick,
}: {
  rounds: DisplayGame[][];
  direction: 'ltr' | 'rtl';
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const ordered = direction === 'rtl' ? [...rounds].reverse() : rounds;
  return (
    <div className="relative flex" style={{ height: REGION_H, gap: 1 }}>
      {ordered.map((games, col) => {
        const roundIdx = direction === 'rtl' ? 3 - col : col;
        return (
          <RoundCol
            key={col}
            games={games}
            roundIdx={roundIdx}
            onHover={onHover}
            setPos={setPos}
            onClick={onClick}
          />
        );
      })}
    </div>
  );
}

function RoundCol({
  games, roundIdx, onHover, setPos, onClick,
}: {
  games: DisplayGame[];
  roundIdx: number;
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  return (
    <div className="relative flex-shrink-0" style={{ width: ROUND_W, height: REGION_H }}>
      {games.map((game, gi) => (
        <div
          key={game.id}
          className="absolute"
          style={{ top: gameTopPx(roundIdx, gi), width: ROUND_W }}
        >
          <GameCard game={game} onHover={onHover} setPos={setPos} onClick={onClick} />
        </div>
      ))}
    </div>
  );
}

function GameCard({
  game, onHover, setPos, onClick,
}: {
  game: DisplayGame;
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const { teamA, teamB } = game;
  if (!teamA || !teamB) return null;
  return (
    <div
      className="overflow-hidden"
      style={{
        height: SLOT_H * 2,
        border: '1px solid #0d1e30',
        background: '#050a14',
      }}
    >
      <TeamRow
        team={teamA}
        isWinner={game.winner?.id === teamA.id}
        prob={game.winProbA}
        consensus={game.teamAConsensus}
        onHover={onHover}
        setPos={setPos}
        onClick={onClick}
      />
      <div style={{ height: 1, background: '#0d1e30' }} />
      <TeamRow
        team={teamB}
        isWinner={game.winner?.id === teamB.id}
        prob={1 - game.winProbA}
        consensus={game.teamBConsensus}
        onHover={onHover}
        setPos={setPos}
        onClick={onClick}
      />
    </div>
  );
}

function TeamRow({
  team, isWinner, prob, consensus, onHover, setPos, onClick,
}: {
  team: Team;
  isWinner: boolean;
  prob: number;
  consensus?: number;
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const barColor = winProbColor(prob);
  const pctDisplay = consensus !== undefined
    ? `${consensus.toFixed(0)}%`
    : `${Math.round(prob * 100)}%`;

  return (
    <div
      className="relative flex items-center overflow-hidden"
      style={{
        height: SLOT_H,
        background: isWinner ? '#091626' : '#050a14',
        borderLeft: isWinner ? `2px solid ${barColor}` : '2px solid transparent',
        cursor: onClick ? 'pointer' : 'default',
      }}
      onMouseEnter={e => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setPos({ x: rect.right + 8, y: rect.top });
        onHover(team);
      }}
      onMouseLeave={() => onHover(null)}
      onClick={() => onClick?.(team)}
    >
      {/* Probability bar bg */}
      <div
        className="absolute left-0 top-0 bottom-0 opacity-[0.07]"
        style={{ width: `${Math.round(prob * 100)}%`, background: barColor }}
      />

      {/* Seed */}
      <span
        className="flex-shrink-0 text-center font-bold tabular-nums"
        style={{ fontSize: 9, color: seedColor(team.seed), width: 18, paddingLeft: 2 }}
      >
        {team.seed}
      </span>

      {/* Name */}
      <span
        className="flex-1 truncate"
        style={{
          fontSize: 10,
          color: isWinner ? '#cbd5e1' : '#475569',
          paddingLeft: 3,
          paddingRight: 2,
          fontWeight: isWinner ? 600 : 400,
          letterSpacing: '0.02em',
          textDecoration: (!isWinner && team.seed <= 8) ? undefined : undefined,
        }}
      >
        {team.name}
      </span>

      {/* Pct */}
      <span
        className="flex-shrink-0 pr-1 tabular-nums font-bold"
        style={{ fontSize: 9, color: consensus !== undefined ? '#f59e0b' : barColor }}
      >
        {pctDisplay}
      </span>
    </div>
  );
}

function CenterColumn({
  bracket, consensus, champion, onHover, setPos, onClick,
}: {
  bracket: DisplayBracket;
  consensus: ConsensusData | null;
  champion: Team | null;
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const halfH = REGION_H;
  const totalH = halfH * 2 + 20;

  return (
    <div
      className="flex flex-col items-center justify-center"
      style={{
        minWidth: 230,
        paddingTop: halfH / 2 - SLOT_H * 2,
        height: totalH,
      }}
    >
      <div className="text-[9px] tracking-[0.3em] text-slate-700 mb-2 uppercase">Final Four</div>

      <FFGameBlock game={bracket.finalFour[0]} label="EAST · SOUTH" onHover={onHover} setPos={setPos} onClick={onClick} />

      <div className="my-3 w-full">
        <ChampBlock game={bracket.championship} champion={champion} consensus={consensus} onHover={onHover} setPos={setPos} onClick={onClick} />
      </div>

      <FFGameBlock game={bracket.finalFour[1]} label="MIDWEST · WEST" onHover={onHover} setPos={setPos} onClick={onClick} />
    </div>
  );
}

function FFGameBlock({
  game, label, onHover, setPos, onClick,
}: {
  game: DisplayGame;
  label: string;
  onHover: (t: Team | null) => void;
  setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const { teamA, teamB, winner } = game;
  if (!teamA || !teamB) return null;
  return (
    <div style={{ width: 210 }}>
      <div className="text-[8px] tracking-[0.2em] text-slate-700 mb-1">{label}</div>
      <div style={{ border: '1px solid #1e3a5f', background: '#050e1c' }}>
        <FFRow
          team={teamA} isWinner={winner?.id === teamA.id}
          prob={game.winProbA} consensus={game.teamAConsensus}
          onHover={onHover} setPos={setPos} onClick={onClick}
        />
        <div style={{ height: 1, background: '#1e3a5f' }} />
        <FFRow
          team={teamB} isWinner={winner?.id === teamB.id}
          prob={1 - game.winProbA} consensus={game.teamBConsensus}
          onHover={onHover} setPos={setPos} onClick={onClick}
        />
      </div>
    </div>
  );
}

function FFRow({
  team, isWinner, prob, consensus, onHover, setPos, onClick,
}: {
  team: Team; isWinner: boolean; prob: number; consensus?: number;
  onHover: (t: Team | null) => void; setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  return (
    <div
      className="flex items-center gap-2 px-3"
      style={{
        height: 30,
        background: isWinner ? '#0c1f38' : '#050e1c',
        borderLeft: isWinner ? `3px solid ${winProbColor(prob)}` : '3px solid transparent',
        cursor: onClick ? 'pointer' : 'default',
      }}
      onMouseEnter={e => {
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setPos({ x: r.right + 8, y: r.top });
        onHover(team);
      }}
      onMouseLeave={() => onHover(null)}
      onClick={() => onClick?.(team)}
    >
      <span className="font-bold tabular-nums" style={{ fontSize: 10, color: seedColor(team.seed), width: 16 }}>
        {team.seed}
      </span>
      <span className="flex-1 truncate text-xs font-medium" style={{ color: isWinner ? '#e2e8f0' : '#475569' }}>
        {team.name}
      </span>
      <span className="text-xs font-bold tabular-nums" style={{ color: winProbColor(prob) }}>
        {consensus !== undefined ? `${consensus.toFixed(0)}%` : `${Math.round(prob * 100)}%`}
      </span>
    </div>
  );
}

function ChampBlock({
  game, champion, consensus, onHover, setPos, onClick,
}: {
  game: DisplayGame; champion: Team | null; consensus: ConsensusData | null;
  onHover: (t: Team | null) => void; setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  const { teamA, teamB, winner } = game;
  if (!teamA || !teamB) return null;
  return (
    <div
      style={{
        border: '2px solid #f59e0b',
        background: '#08060a',
        boxShadow: '0 0 24px #f59e0b22, inset 0 0 12px #f59e0b08',
      }}
    >
      {/* Header */}
      <div
        className="text-center py-1 text-[9px] font-bold tracking-[0.3em] uppercase"
        style={{ background: '#0f0900', color: '#f59e0b', borderBottom: '1px solid #f59e0b44' }}
      >
        National Championship
      </div>

      {[teamA, teamB].map(t => (
        <div key={t.id}>
          <ChampRow
            team={t}
            isWinner={winner?.id === t.id}
            prob={t === teamA ? game.winProbA : 1 - game.winProbA}
            consensus={t === teamA ? game.teamAConsensus : game.teamBConsensus}
            onHover={onHover}
            setPos={setPos}
            onClick={onClick}
          />
          {t === teamA && <div style={{ height: 1, background: '#f59e0b22' }} />}
        </div>
      ))}

      {winner && (
        <div
          className="text-center py-1.5 text-xs font-black tracking-[0.15em] uppercase"
          style={{ background: '#100800', color: '#fbbf24', borderTop: '1px solid #f59e0b44' }}
        >
          CHAMPION: {winner.name}
          {consensus && (
            <span className="ml-2 text-amber-600">
              ({(((consensus.championFreq.get(winner.id) ?? 0) / consensus.totalSims) * 100).toFixed(1)}%)
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function ChampRow({
  team, isWinner, prob, consensus, onHover, setPos, onClick,
}: {
  team: Team; isWinner: boolean; prob: number; consensus?: number;
  onHover: (t: Team | null) => void; setPos: (p: { x: number; y: number }) => void;
  onClick?: (t: Team) => void;
}) {
  return (
    <div
      className="flex items-center gap-2 px-3"
      style={{
        height: 36,
        background: isWinner ? '#160c00' : '#08060a',
        borderLeft: isWinner ? '3px solid #f59e0b' : '3px solid transparent',
        cursor: onClick ? 'pointer' : 'default',
      }}
      onMouseEnter={e => {
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setPos({ x: r.right + 8, y: r.top });
        onHover(team);
      }}
      onMouseLeave={() => onHover(null)}
      onClick={() => onClick?.(team)}
    >
      <span className="font-bold tabular-nums" style={{ fontSize: 11, color: seedColor(team.seed), width: 18 }}>
        {team.seed}
      </span>
      <span className="flex-1 truncate font-bold" style={{ fontSize: 12, color: isWinner ? '#fbbf24' : '#475569' }}>
        {team.name}
      </span>
      <span className="text-sm font-bold tabular-nums" style={{ color: isWinner ? '#f59e0b' : '#334155' }}>
        {consensus !== undefined ? `${consensus.toFixed(1)}%` : `${Math.round(prob * 100)}%`}
      </span>
    </div>
  );
}

// ── Player tooltip ────────────────────────────────────────────────────────

function PlayerTooltip({ team, x, y }: { team: Team; x: number; y: number }) {
  const players = team.roster ?? [];
  const starters = players.filter(p => p.isStarter).slice(0, 5);
  const bench = players.filter(p => !p.isStarter).slice(0, 3);

  return (
    <div
      className="fixed z-[9999] pointer-events-none"
      style={{
        left: Math.min(x, window.innerWidth - 260),
        top: Math.max(8, Math.min(y - 20, window.innerHeight - 320)),
        width: 248,
        border: '1px solid #1e3a5f',
        background: '#040d1a',
        fontFamily: 'monospace',
      }}
    >
      {/* Team header */}
      <div
        className="px-2 py-1 flex items-center justify-between"
        style={{ background: '#050e1c', borderBottom: '1px solid #1e3a5f' }}
      >
        <div>
          <span className="text-[9px] tracking-widest text-slate-600 uppercase">{team.region} • #{team.seed}</span>
          <div className="text-xs font-bold text-slate-200">{team.name}</div>
        </div>
        <div className="text-right">
          <div className="text-[9px] text-slate-600">NET</div>
          <div className="text-sm font-black text-amber-400">{team.netRanking}</div>
        </div>
      </div>

      {/* Team stats mini-row */}
      <div className="grid grid-cols-4 gap-0 border-b border-[#1e3a5f]">
        {[
          { l: 'ADJOE', v: team.adjOE.toFixed(1) },
          { l: 'ADJDE', v: team.adjDE.toFixed(1) },
          { l: 'TEMPO', v: team.adjTempo.toFixed(1) },
          { l: 'W-L', v: `${team.wins}-${team.losses}` },
        ].map(s => (
          <div key={s.l} className="text-center py-1 border-r border-[#1e3a5f] last:border-r-0">
            <div className="text-[7px] text-slate-700 uppercase">{s.l}</div>
            <div className="text-[10px] font-bold text-slate-400">{s.v}</div>
          </div>
        ))}
      </div>

      {/* Players */}
      {players.length > 0 ? (
        <>
          <PlayerHeader />
          {starters.map(p => <PlayerRow key={p.id} player={p} starter />)}
          {bench.length > 0 && (
            <>
              <div className="text-[8px] text-slate-700 px-2 py-0.5 bg-[#040a14] uppercase tracking-widest">Bench</div>
              {bench.map(p => <PlayerRow key={p.id} player={p} starter={false} />)}
            </>
          )}
        </>
      ) : (
        <div className="px-2 py-2 text-[10px] text-slate-600">
          Loading player data...
        </div>
      )}
    </div>
  );
}

function PlayerHeader() {
  return (
    <div className="grid px-2 py-0.5 bg-[#030810]" style={{ gridTemplateColumns: '1fr 28px 28px 28px 28px 30px' }}>
      <span className="text-[7px] text-slate-700 uppercase">PLAYER</span>
      <span className="text-[7px] text-slate-700 text-right">PTS</span>
      <span className="text-[7px] text-slate-700 text-right">REB</span>
      <span className="text-[7px] text-slate-700 text-right">AST</span>
      <span className="text-[7px] text-slate-700 text-right">USG</span>
      <span className="text-[7px] text-slate-700 text-right">TS%</span>
    </div>
  );
}

function PlayerRow({ player: p, starter }: { player: Player; starter: boolean }) {
  return (
    <div
      className="grid px-2 py-0.5"
      style={{
        gridTemplateColumns: '1fr 28px 28px 28px 28px 30px',
        background: starter ? '#050e1c' : '#040a14',
        borderBottom: '1px solid #0a1628',
      }}
    >
      <div className="truncate">
        <span className="text-[9px] font-bold" style={{ color: starter ? '#94a3b8' : '#475569' }}>
          {p.name.split(' ').slice(-1)[0]}
        </span>
        <span className="text-[8px] text-slate-700 ml-1">{p.position}</span>
      </div>
      <span className="text-[9px] text-right tabular-nums" style={{ color: p.ppg >= 15 ? '#f59e0b' : '#64748b' }}>
        {p.ppg.toFixed(1)}
      </span>
      <span className="text-[9px] text-right tabular-nums text-slate-600">{p.rpg.toFixed(1)}</span>
      <span className="text-[9px] text-right tabular-nums text-slate-600">{p.apg.toFixed(1)}</span>
      <span className="text-[9px] text-right tabular-nums text-slate-700">
        {(p.usageRate * 100).toFixed(0)}%
      </span>
      <span
        className="text-[9px] text-right tabular-nums"
        style={{ color: p.trueShootingPct > 0.58 ? '#22c55e' : p.trueShootingPct > 0.52 ? '#64748b' : '#ef4444' }}
      >
        {(p.trueShootingPct * 100).toFixed(0)}%
      </span>
    </div>
  );
}
