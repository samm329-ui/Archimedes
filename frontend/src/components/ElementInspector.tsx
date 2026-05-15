"use client";

import { useState } from "react";

interface ElementInspectorProps {
  elements: Array<{
    id: string;
    type: string;
    confidence: number;
    motion?: string[];
  }>;
}

export default function ElementInspector({ elements }: ElementInspectorProps) {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Element Inspector</h2>
      <div className="space-y-2 max-h-96 overflow-auto">
        {elements.map((el) => (
          <div
            key={el.id}
            className={`p-3 rounded-lg cursor-pointer transition-colors ${
              selected === el.id ? "bg-blue-600" : "bg-slate-700 hover:bg-slate-600"
            }`}
            onClick={() => setSelected(el.id)}
          >
            <div className="flex justify-between items-start">
              <div>
                <div className="font-medium text-white">{el.id}</div>
                <div className="text-sm text-slate-400">{el.type}</div>
              </div>
              <div className="text-sm font-medium text-emerald-400">
                {Math.round(el.confidence * 100)}%
              </div>
            </div>
            {el.motion && el.motion.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {el.motion.map((m, i) => (
                  <span key={i} className="text-xs bg-slate-600 px-2 py-0.5 rounded text-slate-300">
                    {m}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}