'use client';

import { SLOT_H, ROUND_W, COL_GAP, REGION_H, gameTopPx, seedColor, winProbColor } from '@/lib/bracket';
import type { DisplayBracket, DisplayGame } from '@/lib/simulation';
import type { Team } from '@/lib/types';
import type { ConsensusData } from '@/lib/simulation';

interface Props {
  bracket: DisplayBracket;
  consensus: ConsensusData | null;
  champion: Team | null;
}

export default function BracketDisplay({ bracket, consensus, champion }: Props) {
  return (
    <div className="w-full overflow-x-auto pb-6">
      <div
        className="inline-flex items-start gap-0 min-w-max mx-auto"
        style={{ paddingTop: '2rem', paddingBottom: '2rem' }}
      >
        {/* LEFT HALF: East (top) + South (bottom) → R64 to E8 */}
        <div className="flex flex-col gap-4">
          <RegionLabel name="EAST" align="left" />
          <RegionHalf rounds={bracket.east} direction="ltr" regionIdx={0} />
          <div style={{ height: 16 }} />
          <RegionLabel name="SOUTH" align="left" />
          <RegionHalf rounds={bracket.south} direction="ltr" regionIdx={1} />
        </div>

        {/* CENTER: Final Four + Championship */}
        <div className="flex flex-col items-center justify-center" style={{ minWidth: 240, paddingTop: (REGION_H / 2) - SLOT_H }}>
          <div className="text-xs font-bold text-amber-500 uppercase tracking-[0.2em] mb-3">Final Four</div>

          <FFGame game={bracket.finalFour[0]} consensus={consensus} label="East vs South" />
          <div className="my-4 text-center">
            <div className="text-xs text-slate-600 uppercase tracking-widest mb-2">National Championship</div>
            <ChampionshipGame game={bracket.championship} consensus={consensus} champion={champion} />
          </div>
          <FFGame game={bracket.finalFour[1]} consensus={consensus} label="Midwest vs West" />
        </div>

        {/* RIGHT HALF: Midwest (top) + West (bottom) → E8 to R64 */}
        <div className="flex flex-col gap-4">
          <RegionLabel name="MIDWEST" align="right" />
          <RegionHalf rounds={bracket.midwest} direction="rtl" regionIdx={2} />
          <div style={{ height: 16 }} />
          <RegionLabel name="WEST" align="right" />
          <RegionHalf rounds={bracket.west} direction="rtl" regionIdx={3} />
        </div>
      </div>
    </div>
  );
}

function RegionLabel({ name, align }: { name: string; align: 'left' | 'right' }) {
  return (
    <div className={`text-xs font-bold text-slate-500 uppercase tracking-[0.25em] ${align === 'right' ? 'text-right' : 'text-left'}`}>
      {name}
    </div>
  );
}

// Renders 4 rounds of a region. Direction: ltr = R64 on left → E8 on right, rtl = reversed.
function RegionHalf({
  rounds,
  direction,
  regionIdx,
}: {
  rounds: DisplayGame[][];
  direction: 'ltr' | 'rtl';
  regionIdx: number;
}) {
  const orderedRounds = direction === 'rtl' ? [...rounds].reverse() : rounds;

  return (
    <div
      className="relative flex gap-[2px]"
      style={{ height: REGION_H }}
    >
      {orderedRounds.map((games, displayColIdx) => {
        const actualRoundIdx = direction === 'rtl' ? 3 - displayColIdx : displayColIdx;
        return (
          <RoundColumn
            key={actualRoundIdx}
            games={games}
            roundIdx={actualRoundIdx}
            direction={direction}
          />
        );
      })}
    </div>
  );
}

function RoundColumn({
  games,
  roundIdx,
  direction,
}: {
  games: DisplayGame[];
  roundIdx: number;
  direction: 'ltr' | 'rtl';
}) {
  return (
    <div
      className="relative flex-shrink-0"
      style={{ width: ROUND_W, height: REGION_H }}
    >
      {games.map((game, gi) => {
        const top = gameTopPx(roundIdx, gi);
        return (
          <div
            key={game.id}
            className="absolute"
            style={{ top, width: ROUND_W, height: SLOT_H * 2 }}
          >
            <GameCard game={game} compact />
          </div>
        );
      })}
    </div>
  );
}

function GameCard({ game, compact = false }: { game: DisplayGame; compact?: boolean }) {
  const { teamA, teamB, winner } = game;
  if (!teamA || !teamB) return null;

  return (
    <div
      className="flex flex-col overflow-hidden rounded border border-[#172236]"
      style={{ height: SLOT_H * 2, width: '100%' }}
    >
      <TeamRow team={teamA} isWinner={winner?.id === teamA.id} probWin={game.winProbA} consensus={game.teamAConsensus} />
      <div className="border-t border-[#172236]" />
      <TeamRow team={teamB} isWinner={winner?.id === teamB.id} probWin={1 - game.winProbA} consensus={game.teamBConsensus} />
    </div>
  );
}

function TeamRow({
  team,
  isWinner,
  probWin,
  consensus,
}: {
  team: Team;
  isWinner: boolean;
  probWin: number;
  consensus?: number;
}) {
  const bg = isWinner ? '#0f1e35' : '#080e1c';
  const textOpacity = isWinner ? '1' : '0.55';
  const barColor = winProbColor(probWin);

  return (
    <div
      className="relative flex items-center overflow-hidden"
      style={{ height: SLOT_H, background: bg, cursor: 'default' }}
    >
      {/* Win probability bar (background) */}
      <div
        className="absolute left-0 top-0 bottom-0 opacity-10"
        style={{ width: `${Math.round(probWin * 100)}%`, background: barColor }}
      />

      {/* Seed badge */}
      <div
        className="flex-shrink-0 w-5 text-center font-bold font-mono"
        style={{ fontSize: 9, color: seedColor(team.seed), marginLeft: 4 }}
      >
        {team.seed}
      </div>

      {/* Name */}
      <div
        className="flex-1 truncate font-medium"
        style={{
          fontSize: 10,
          opacity: textOpacity,
          color: isWinner ? '#e2e8f0' : '#94a3b8',
          paddingLeft: 4,
          paddingRight: 2,
          letterSpacing: '0.01em',
        }}
      >
        {team.name}
      </div>

      {/* Consensus % or win prob */}
      <div className="flex-shrink-0 pr-1 text-right">
        {consensus !== undefined ? (
          <span style={{ fontSize: 9, color: '#f59e0b', fontWeight: 700, fontFamily: 'monospace' }}>
            {consensus.toFixed(0)}%
          </span>
        ) : (
          <span style={{ fontSize: 9, color: barColor, fontFamily: 'monospace' }}>
            {Math.round(probWin * 100)}%
          </span>
        )}
      </div>

      {/* Winner indicator */}
      {isWinner && (
        <div
          className="absolute right-0 top-0 bottom-0 w-0.5"
          style={{ background: barColor }}
        />
      )}
    </div>
  );
}

function FFGame({
  game,
  consensus,
  label,
}: {
  game: DisplayGame;
  consensus: ConsensusData | null;
  label: string;
}) {
  const { teamA, teamB, winner } = game;
  if (!teamA || !teamB) return null;

  return (
    <div className="mb-2">
      <div className="text-xs text-slate-600 uppercase tracking-widest mb-1 text-center">{label}</div>
      <div
        className="rounded-lg border border-[#1e3058] overflow-hidden"
        style={{ width: 200 }}
      >
        <FFTeamRow team={teamA} isWinner={winner?.id === teamA.id} prob={game.winProbA} consensus={game.teamAConsensus} />
        <div className="border-t border-[#1e3058]" />
        <FFTeamRow team={teamB} isWinner={winner?.id === teamB.id} prob={1 - game.winProbA} consensus={game.teamBConsensus} />
      </div>
    </div>
  );
}

function FFTeamRow({
  team,
  isWinner,
  prob,
  consensus,
}: {
  team: Team;
  isWinner: boolean;
  prob: number;
  consensus?: number;
}) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-2"
      style={{ background: isWinner ? '#0f2040' : '#080e1c' }}
    >
      <span
        className="font-bold font-mono text-xs w-4 text-center"
        style={{ color: seedColor(team.seed) }}
      >
        {team.seed}
      </span>
      <span
        className="flex-1 text-sm font-medium truncate"
        style={{ color: isWinner ? '#e2e8f0' : '#64748b' }}
      >
        {team.name}
      </span>
      <span className="text-xs font-mono" style={{ color: winProbColor(prob) }}>
        {consensus !== undefined ? `${consensus.toFixed(0)}%` : `${Math.round(prob * 100)}%`}
      </span>
    </div>
  );
}

function ChampionshipGame({
  game,
  consensus,
  champion,
}: {
  game: DisplayGame;
  consensus: ConsensusData | null;
  champion: Team | null;
}) {
  const { teamA, teamB, winner } = game;
  if (!teamA || !teamB) return null;

  return (
    <div
      className="rounded-xl border-2 overflow-hidden"
      style={{
        width: 220,
        borderColor: '#f59e0b',
        boxShadow: '0 0 30px #f59e0b44, 0 0 60px #f59e0b22',
      }}
    >
      <div className="text-center py-1.5 text-xs font-bold text-amber-400 uppercase tracking-widest" style={{ background: '#1a0e00' }}>
        National Championship
      </div>
      <ChampTeamRow team={teamA} isWinner={winner?.id === teamA.id} prob={game.winProbA} consensus={game.teamAConsensus} />
      <div className="border-t border-[#3d2200]" />
      <ChampTeamRow team={teamB} isWinner={winner?.id === teamB.id} prob={1 - game.winProbA} consensus={game.teamBConsensus} />
      {winner && (
        <div
          className="text-center py-2 text-xs font-bold uppercase tracking-widest"
          style={{ background: '#1a0e00', color: '#f59e0b' }}
        >
          🏆 {winner.name}
        </div>
      )}
    </div>
  );
}

function ChampTeamRow({
  team,
  isWinner,
  prob,
  consensus,
}: {
  team: Team;
  isWinner: boolean;
  prob: number;
  consensus?: number;
}) {
  return (
    <div
      className="flex items-center gap-2 px-3 py-2.5"
      style={{ background: isWinner ? '#1a1000' : '#08090e' }}
    >
      <span
        className="font-bold font-mono text-sm w-5 text-center"
        style={{ color: seedColor(team.seed) }}
      >
        {team.seed}
      </span>
      <span
        className="flex-1 text-sm font-bold truncate"
        style={{ color: isWinner ? '#fbbf24' : '#475569' }}
      >
        {team.name}
      </span>
      <span
        className="text-sm font-bold font-mono"
        style={{ color: isWinner ? '#f59e0b' : '#64748b' }}
      >
        {consensus !== undefined ? `${consensus.toFixed(0)}%` : `${Math.round(prob * 100)}%`}
      </span>
    </div>
  );
}
