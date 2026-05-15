"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { PipelineResult } from "@/types/pipeline";
import FrameViewer from "@/components/FrameViewer";
import Timeline from "@/components/Timeline";
import JSONViewer from "@/components/JSONViewer";
import ElementInspector from "@/components/ElementInspector";
import SceneInspector from "@/components/SceneInspector";
import ValidationPanel from "@/components/ValidationPanel";
import ConfidencePanel from "@/components/ConfidencePanel";
import HypothesisPanel from "@/components/HypothesisPanel";
import DiffViewer from "@/components/DiffViewer";
import Link from "next/link";

type Panel = "elements" | "scenes" | "validation" | "confidence" | "hypotheses" | "diff" | "json";

const PANELS: { key: Panel; label: string; title: string }[] = [
  { key: "elements",   label: "Elements",    title: "Element Inspector" },
  { key: "scenes",     label: "Scenes",      title: "Scene Inspector" },
  { key: "validation", label: "Validation",  title: "Validation Panel" },
  { key: "confidence", label: "Confidence",  title: "Confidence Panel" },
  { key: "hypotheses", label: "Hypotheses",  title: "Hypothesis Comparison" },
  { key: "diff",       label: "Diff",        title: "Render vs Source" },
  { key: "json",       label: "JSON",        title: "JSON Viewer" },
];

export default function InspectionPage() {
  const router = useRouter();
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activePanel, setActivePanel] = useState<Panel>("elements");

  useEffect(() => {
    const raw = sessionStorage.getItem("pipelineResult");
    if (!raw) { router.push("/upload"); return; }
    try { setResult(JSON.parse(raw)); }
    catch { router.push("/upload"); }
  }, [router]);

  if (!result?.template) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { template, errors, diagnostics, validation_report } = result;
  const { meta, elements, scenes, validation } = template;

  const statusColor =
    result.status === "approved" ? "text-green-400" :
    result.status === "needs_refinement" ? "text-yellow-400" : "text-red-400";

  return (
    <div className="h-screen bg-gray-950 text-white flex flex-col overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center gap-3 px-4 py-2 border-b border-gray-800 bg-gray-900">
        <Link href="/result" className="text-gray-400 hover:text-white text-sm transition-colors">
          ← Result
        </Link>
        <div className="w-px h-4 bg-gray-700" />
        <span className="text-sm font-semibold text-gray-300 truncate max-w-[180px]">
          {meta.filename}
        </span>
        <span className="text-xs text-gray-600 font-mono hidden md:block">
          {meta.width}×{meta.height} · {meta.fps.toFixed(1)}fps · {meta.frame_count}f
        </span>
        <span className={`text-xs font-semibold ${statusColor}`}>
          {result.status.replace(/_/g, " ")}
        </span>

        <div className="flex-1" />

        {/* Panel tabs */}
        <nav className="flex items-center gap-0.5 bg-gray-800 rounded-lg p-0.5">
          {PANELS.map(p => (
            <button
              key={p.key}
              onClick={() => setActivePanel(p.key)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors
                ${activePanel === p.key
                  ? "bg-indigo-600 text-white shadow"
                  : "text-gray-400 hover:text-gray-200"}`}
            >
              {p.label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: viewer + timeline */}
        <div className="flex flex-col flex-1 min-w-0 overflow-hidden p-3 gap-2.5">
          {/* Canvas viewer */}
          <div className="flex-shrink-0">
            <FrameViewer
              frameIndex={currentFrame}
              meta={meta}
              elements={elements}
              scenes={scenes}
              width={640}
              height={360}
            />
          </div>

          {/* Timeline */}
          <Timeline
            meta={meta}
            elements={elements}
            scenes={scenes}
            currentFrame={currentFrame}
            onFrameChange={setCurrentFrame}
          />

          {/* Scrubber */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => setCurrentFrame(f => Math.max(0, f - 1))}
              className="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-sm transition-colors"
            >◀</button>
            <input
              type="range" min={0} max={meta.frame_count - 1}
              value={currentFrame}
              onChange={e => setCurrentFrame(Number(e.target.value))}
              className="flex-1 accent-indigo-500"
            />
            <button
              onClick={() => setCurrentFrame(f => Math.min(meta.frame_count - 1, f + 1))}
              className="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-sm transition-colors"
            >▶</button>
            <span className="text-xs font-mono text-gray-500 w-24 text-right">
              {currentFrame} / {meta.frame_count - 1}
            </span>
          </div>

          {/* Stats bar */}
          <div className="flex gap-4 shrink-0 px-1">
            {[
              { label: "Elements", value: elements.length },
              { label: "Scenes",   value: scenes.length },
              { label: "Errors",   value: errors.length },
              { label: "SSIM",     value: validation.ssim_score != null
                  ? `${(validation.ssim_score * 100).toFixed(1)}%` : "—" },
            ].map(({ label, value }) => (
              <div key={label} className="text-center">
                <p className="text-xs text-gray-600">{label}</p>
                <p className="text-sm font-mono font-bold text-gray-300">{value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Right: active panel */}
        <aside className="w-[400px] shrink-0 border-l border-gray-800 flex flex-col overflow-hidden p-3">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider mb-2 shrink-0">
            {PANELS.find(p => p.key === activePanel)?.title}
          </p>
          <div className="flex-1 overflow-y-auto">
            {activePanel === "elements" && (
              <ElementInspector
                elements={elements}
                selectedId={selectedId}
                onSelect={setSelectedId}
                currentFrame={currentFrame}
              />
            )}
            {activePanel === "scenes" && (
              <SceneInspector
                scenes={scenes}
                meta={meta}
                elements={elements}
                currentFrame={currentFrame}
                onSeek={setCurrentFrame}
              />
            )}
            {activePanel === "validation" && (
              <ValidationPanel
                report={validation_report}
                validation={validation}
                errors={errors}
              />
            )}
            {activePanel === "confidence" && (
              <ConfidencePanel
                elements={elements}
                diagnostics={diagnostics}
              />
            )}
            {activePanel === "hypotheses" && (
              <HypothesisPanel
                elements={elements}
                selectedId={selectedId}
              />
            )}
            {activePanel === "diff" && (
              <DiffViewer
                template={template}
                validation={validation}
              />
            )}
            {activePanel === "json" && (
              <JSONViewer
                data={template}
                title="Full Template JSON"
                maxHeight="calc(100vh - 200px)"
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
