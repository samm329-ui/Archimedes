"use client";

interface Hypothesis {
  id: string;
  type: string;
  confidence: number;
  reasoning?: string;
}

interface HypothesisPanelProps {
  elementId: string;
  hypotheses: Hypothesis[];
  selected?: string;
  onSelect: (id: string) => void;
}

export default function HypothesisPanel({ elementId, hypotheses, selected, onSelect }: HypothesisPanelProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-2">Motion Hypotheses</h2>
      <div className="text-sm text-slate-400 mb-4">Element: {elementId}</div>
      <div className="space-y-2 max-h-64 overflow-auto">
        {hypotheses.map((h) => (
          <div
            key={h.id}
            className={`p-3 rounded-lg cursor-pointer transition-colors ${
              selected === h.id ? "bg-amber-600" : "bg-slate-700 hover:bg-slate-600"
            }`}
            onClick={() => onSelect(h.id)}
          >
            <div className="flex justify-between items-start">
              <div className="font-medium text-white">{h.type}</div>
              <div className="text-sm font-medium text-amber-400">
                {Math.round(h.confidence * 100)}%
              </div>
            </div>
            {h.reasoning && (
              <div className="mt-1 text-xs text-slate-400">{h.reasoning}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}