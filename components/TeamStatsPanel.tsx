import React from 'react';
import type { Team } from '@/lib/types';
import type { ConsensusData } from '@/lib/simulation';

interface TeamStatsPanelProps {
  team: Team | null;
  onClose: () => void;
  consensus: ConsensusData | null;
}

const TeamStatsPanel: React.FC<TeamStatsPanelProps> = ({ team, onClose, consensus }) => {
  if (!team) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
          style={{ fontSize: 18, fontWeight: 700 }}
        >
          ×
        </button>
        <h2 className="text-xl font-bold mb-2">{team.name} (#{team.seed} {team.region})</h2>
        <div className="mb-4 text-sm text-gray-700">
          <div>Record: {team.wins}-{team.losses}</div>
          <div>NET Ranking: {team.netRanking}</div>
          <div>Coach Tournament Wins: {team.coachTourneyWins}</div>
          <div>Adj. Off Eff: {team.adjOE.toFixed(1)} | Adj. Def Eff: {team.adjDE.toFixed(1)}</div>
          <div>Efficiency Margin: {(team.adjOE - team.adjDE).toFixed(1)}</div>
        </div>
        <div className="mb-4">
          <h3 className="font-semibold text-sm mb-1">Roster</h3>
          <ul className="text-xs text-gray-600">
            {team.roster && team.roster.length > 0 ? (
              team.roster.map((player, idx) => (
                <li key={player.id || idx}>{player.name} — {player.position} (BPM: {player.bpm.toFixed(2)})</li>
              ))
            ) : (
              <li>No roster data available.</li>
            )}
          </ul>
        </div>
        {consensus && (
          <div className="mt-4 text-xs text-gray-700">
            <div>Consensus Final Four Frequency: {((consensus.ffFreq.get(team.id) ?? 0) / consensus.totalSims * 100).toFixed(1)}%</div>
            <div>Consensus Champion Frequency: {((consensus.championFreq.get(team.id) ?? 0) / consensus.totalSims * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TeamStatsPanel;
