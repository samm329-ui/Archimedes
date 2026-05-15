"use client";

interface ConfidencePanelProps {
  stage: string;
  confidence: number;
  timing?: {
    stage: string;
    duration_ms: number;
  }[];
}

export default function ConfidencePanel({ stage, confidence, timing }: ConfidencePanelProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Pipeline Confidence</h2>
      <div className="mb-4">
        <div className="text-2xl font-bold text-emerald-400">{stage}</div>
        <div className="text-slate-400">Current Stage</div>
      </div>
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-slate-300">Confidence</span>
          <span className="text-white">{Math.round(confidence * 100)}%</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>
      {timing && timing.length > 0 && (
        <div className="mt-4">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Timing Profile</h3>
          <div className="space-y-1">
            {timing.map((t, i) => (
              <div key={i} className="flex justify-between text-xs text-slate-400">
                <span>{t.stage}</span>
                <span>{t.duration_ms}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}