"use client";

interface DiffViewerProps {
  original: string;
  reconstructed: string;
  ssimScore?: number;
}

export default function DiffViewer({ original, reconstructed, ssimScore }: DiffViewerProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Render Comparison</h2>
      {ssimScore !== undefined && (
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-300">SSIM Score</span>
            <span className="text-white">{ssimScore.toFixed(3)}</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${ssimScore * 100}%` }}
            />
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm text-slate-400 mb-2">Original</h3>
          <pre className="bg-slate-900 p-2 rounded text-xs text-slate-300 overflow-auto max-h-48">
            {original || "No data"}
          </pre>
        </div>
        <div>
          <h3 className="text-sm text-slate-400 mb-2">Reconstructed</h3>
          <pre className="bg-slate-900 p-2 rounded text-xs text-slate-300 overflow-auto max-h-48">
            {reconstructed || "No data"}
          </pre>
        </div>
      </div>
    </div>
  );
}