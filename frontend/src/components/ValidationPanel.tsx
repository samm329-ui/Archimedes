"use client";

interface ValidationLayer {
  name: string;
  passed: boolean;
  score?: number;
  notes?: string;
}

interface ValidationPanelProps {
  layers: ValidationLayer[];
  overallStatus: "approved" | "needs_refinement" | "rejected" | "pending";
}

export default function ValidationPanel({ layers, overallStatus }: ValidationPanelProps) {
  const statusColors = {
    approved: "bg-emerald-500",
    needs_refinement: "bg-amber-500",
    rejected: "bg-red-500",
    pending: "bg-slate-500",
  };

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Validation Report</h2>
      <div className="mb-4">
        <div className="text-sm text-slate-400">Overall Status</div>
        <div className="flex items-center gap-2 mt-1">
          <div className={`w-3 h-3 rounded-full ${statusColors[overallStatus]}`} />
          <span className="font-medium text-white capitalize">
            {overallStatus.replace("_", " ")}
          </span>
        </div>
      </div>
      <div className="space-y-3">
        {layers.map((layer) => (
          <div key={layer.name} className="bg-slate-700 rounded-lg p-3">
            <div className="flex justify-between items-start mb-1">
              <span className="font-medium text-white">{layer.name}</span>
              <span
                className={`text-sm ${layer.passed ? "text-emerald-400" : "text-red-400"}`}
              >
                {layer.passed ? "PASS" : "FAIL"}
              </span>
            </div>
            {layer.score !== undefined && (
              <div className="text-sm text-slate-400">Score: {layer.score.toFixed(2)}</div>
            )}
            {layer.notes && (
              <div className="text-xs text-slate-500 mt-1">{layer.notes}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}