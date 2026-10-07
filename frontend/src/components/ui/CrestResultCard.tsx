import React from 'react';

export interface CrestResultCardProps {
  name: string;
  country: string;
  league: string;
  queryImageUrl: string;
  resultImageUrl: string;
  similarityScore: number;
  distance: number;
  clubLogoUrl?: string;
}

export const CrestResultCard: React.FC<CrestResultCardProps> = ({
  name,
  country,
  league,
  queryImageUrl,
  resultImageUrl,
  similarityScore,
  distance,
  clubLogoUrl,
}) => {
  const getBadgeClasses = (): string => {
    if (similarityScore >= 85) {
      return 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
    }
    if (similarityScore >= 60) {
      return 'bg-amber-500/10 text-amber-400 border border-amber-500/30';
    }
    return 'bg-zinc-800 text-zinc-400 border border-zinc-700';
  };

  const percentage = Math.round(similarityScore);

  return (
    <div className="rounded-[2rem] bg-[#121214] border border-white/10 hover:border-[#FF5500]/50 transition-all p-5">
      <div className="flex items-start gap-4">
        <div className="relative w-24 h-24 rounded-xl overflow-hidden bg-white/5 flex-shrink-0">
          <img
            src={resultImageUrl || queryImageUrl}
            alt={`Resultado: ${name}`}
            className="w-full h-full object-cover"
          />
          {queryImageUrl && (
            <img
              src={queryImageUrl}
              alt={`Consulta: ${name}`}
              className="absolute bottom-1 right-1 w-8 h-8 rounded-md object-cover border border-white/30"
              title="Imagen de consulta"
            />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-white font-semibold text-lg leading-tight">
              {name}
            </h3>
            <span
              className={`rounded-full px-3 py-1 text-xs font-medium ${getBadgeClasses()}`}
            >
              {percentage}%
            </span>
          </div>
          <p className="text-zinc-400 text-sm">
            {country} · {league}
          </p>
          <div className="flex items-center gap-2 mt-3 text-xs text-zinc-500 font-mono">
            <span>Distancia:</span>{' '}
            <span className="text-zinc-300">{distance.toFixed(4)}</span>
          </div>
        </div>
        {clubLogoUrl && (
          <div className="w-16 h-16 rounded-lg overflow-hidden bg-white/5 flex-shrink-0">
            <img
              src={clubLogoUrl}
              alt={`Logo de ${name}`}
              className="w-full h-full object-cover"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default CrestResultCard;
