"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { PipelineResult } from "@/types/pipeline";
import { STATUS_COLORS, confidenceColor, elementColor } from "@/lib/colors";
import Link from "next/link";

export default function ResultPage() {
  const router = useRouter();
  const [result, setResult] = useState<PipelineResult | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("pipelineResult");
    if (!raw) { router.push("/upload"); return; }
    try { setResult(JSON.parse(raw)); }
    catch { router.push("/upload"); }
  }, [router]);

  if (!result) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { status, template, errors, quality, validation_report } = result;
  const statusCls = STATUS_COLORS[status] ?? STATUS_COLORS.pending;
  const meta = template?.meta;
  const elements = template?.elements ?? [];
  const scenes = template?.scenes ?? [];
  const validation = template?.validation;

  const fatalCount = errors.filter(e => !e.recoverable).length;
  const recoverableCount = errors.filter(e => e.recoverable).length;

  const elementTypeCount: Record<string, number> = {};
  elements.forEach(e => { elementTypeCount[e.type] = (elementTypeCount[e.type] ?? 0) + 1; });

  const meanConf = elements.length > 0
    ? elements.reduce((s, e) => s + e.confidence, 0) / elements.length
    : 0;

  const downloadJSON = () => {
    if (!template) return;
    const blob = new Blob([JSON.stringify(template, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `template_${meta?.filename ?? "output"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      {/* Top bar */}
      <div className="border-b border-gray-800 bg-gray-900 px-6 py-3 flex items-center gap-4">
        <Link href="/upload" className="text-gray-400 hover:text-white text-sm">← New analysis</Link>
        <div className="flex-1" />
        <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${statusCls.bg} ${statusCls.text} ${statusCls.border}`}>
          {status.replace(/_/g, " ").toUpperCase()}
        </span>
        <button
          onClick={downloadJSON}
          disabled={!template}
          className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-colors disabled:opacity-40"
        >
          Download JSON
        </button>
        <Link href="/inspection"
          className="px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-white text-sm font-semibold transition-colors">
          Open Inspector →
        </Link>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        {/* Headline metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Elements", value: String(elements.length), sub: `${Object.keys(elementTypeCount).length} types` },
            { label: "Scenes", value: String(scenes.length) },
            { label: "Mean confidence", value: `${Math.round(meanConf * 100)}%`,
              color: confidenceColor(meanConf) },
            { label: "SSIM score",
              value: validation?.ssim_score != null ? `${(validation.ssim_score * 100).toFixed(1)}%` : "—",
              color: validation?.ssim_score != null ? confidenceColor(validation.ssim_score) : "text-gray-400" },
          ].map(({ label, value, sub, color }) => (
            <div key={label} className="rounded-xl bg-gray-900 border border-gray-700 px-4 py-3">
              <p className="text-xs text-gray-500 mb-0.5">{label}</p>
              <p className={`text-2xl font-bold font-mono ${color ?? "text-white"}`}>{value}</p>
              {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
            </div>
          ))}
        </div>

        {/* Video metadata */}
        {meta && (
          <div className="rounded-xl bg-gray-900 border border-gray-700 p-4">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Video</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: "File", value: meta.filename },
                { label: "Resolution", value: `${meta.width}×${meta.height}` },
                { label: "FPS", value: meta.fps.toFixed(2) },
                { label: "Frames", value: String(meta.frame_count) },
                { label: "Duration", value: `${(meta.duration_ms / 1000).toFixed(2)}s` },
                { label: "Codec", value: meta.codec },
                { label: "Audio", value: meta.has_audio ? "Yes" : "No" },
                { label: "Corrupt", value: meta.is_corrupt ? "⚠ Yes" : "No" },
              ].map(({ label, value }) => (
                <div key={label}>
                  <p className="text-xs text-gray-500">{label}</p>
                  <p className="text-xs font-mono text-gray-200 truncate">{value}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Element type breakdown */}
        {elements.length > 0 && (
          <div className="rounded-xl bg-gray-900 border border-gray-700 p-4">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Element Types</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(elementTypeCount).map(([type, count]) => (
                <div key={type} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-800 border border-gray-700">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: elementColor(type) }} />
                  <span className="text-xs font-mono text-gray-300">{type}</span>
                  <span className="text-xs font-mono text-gray-500">×{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Errors */}
        {errors.length > 0 && (
          <div className="rounded-xl bg-gray-900 border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Pipeline Errors</p>
              <div className="flex gap-2">
                {fatalCount > 0 && <span className="text-xs px-2 py-0.5 rounded bg-red-900/50 text-red-400 border border-red-800">{fatalCount} fatal</span>}
                {recoverableCount > 0 && <span className="text-xs px-2 py-0.5 rounded bg-yellow-900/50 text-yellow-400 border border-yellow-800">{recoverableCount} recovered</span>}
              </div>
            </div>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {errors.slice(0, 20).map((err, i) => (
                <div key={i} className={`flex items-start gap-2 text-xs font-mono py-1 border-b border-gray-800
                  ${!err.recoverable ? "text-red-400" : "text-yellow-400"}`}>
                  <span className="shrink-0 text-gray-600">[{err.stage}]</span>
                  <span className="flex-1">{err.message}</span>
                  <span className="shrink-0 text-gray-600">{err.failure_mode}</span>
                </div>
              ))}
              {errors.length > 20 && (
                <p className="text-xs text-gray-600 pt-1">… and {errors.length - 20} more</p>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
