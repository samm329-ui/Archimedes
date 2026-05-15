"use client";
import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { analyzeVideo } from "@/lib/api";
import type { PipelineResult } from "@/types/pipeline";

type Phase = "idle" | "uploading" | "analyzing" | "done" | "error";

const STAGE_LABELS: Record<string, string> = {
  ingest:            "Ingesting video",
  metadata:          "Extracting metadata",
  scene_detection:   "Detecting scenes",
  frame_sampling:    "Sampling frames",
  detection:         "Detecting elements",
  segmentation:      "Segmenting elements",
  feature_extraction:"Extracting features",
  tracking:          "Tracking elements",
  grouping:          "Grouping elements",
  layering:          "Inferring layer order",
  motion_analysis:   "Analysing motion",
  hypothesis_engine: "Consolidating hypotheses",
  role_assignment:   "Assigning roles",
  schema_builder:    "Building template",
  render_loop:       "Rendering & comparing",
  validator:         "Validating output",
};

export default function UploadPage() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("idle");
  const [stageIndex, setStageIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const stages = Object.values(STAGE_LABELS);

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setPhase("uploading");
    setError(null);

    // Simulate stage progression while API runs
    let si = 0;
    const interval = setInterval(() => {
      si = Math.min(si + 1, stages.length - 1);
      setStageIndex(si);
      setPhase("analyzing");
    }, 1400);

    try {
      const result: PipelineResult = await analyzeVideo(file);
      clearInterval(interval);
      setPhase("done");
      // Store result and navigate
      sessionStorage.setItem("pipelineResult", JSON.stringify(result));
      router.push("/result");
    } catch (err) {
      clearInterval(interval);
      setPhase("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [router, stages.length]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const isActive = phase === "uploading" || phase === "analyzing";

  return (
    <main className="min-h-screen bg-gray-950 flex flex-col items-center justify-center p-8">
      {/* Header */}
      <div className="mb-10 text-center">
        <div className="inline-flex items-center gap-2 mb-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Reverse Motion Compiler
          </h1>
        </div>
        <p className="text-gray-400 text-sm max-w-md">
          Upload a motion-graphics MP4. The system will analyse it frame by frame,
          extract motion primitives, and produce a structured JSON template.
        </p>
      </div>

      {/* Upload zone */}
      {!isActive && phase !== "done" && (
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`w-full max-w-xl rounded-2xl border-2 border-dashed transition-all
            ${dragOver ? "border-indigo-400 bg-indigo-900/20" : "border-gray-700 bg-gray-900"}
            flex flex-col items-center justify-center gap-4 py-16 px-8 cursor-pointer`}
        >
          <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center">
            <svg className="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-white font-semibold mb-1">Drop your video here</p>
            <p className="text-gray-500 text-sm">MP4, MOV, AVI, WebM, MKV · max 500 MB</p>
          </div>
          <label className="cursor-pointer">
            <span className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold transition-colors">
              Browse file
            </span>
            <input type="file" accept=".mp4,.mov,.avi,.webm,.mkv" className="hidden" onChange={onFileInput} />
          </label>
        </div>
      )}

      {/* Progress */}
      {isActive && (
        <div className="w-full max-w-xl rounded-2xl border border-gray-700 bg-gray-900 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse" />
            <p className="text-white font-semibold">Analysing {fileName}</p>
          </div>

          {/* Stage list */}
          <div className="space-y-2 mb-6">
            {Object.entries(STAGE_LABELS).map(([key, label], i) => {
              const done = i < stageIndex;
              const active = i === stageIndex;
              return (
                <div key={key} className="flex items-center gap-2.5">
                  <div className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 text-xs
                    ${done ? "bg-green-600" : active ? "bg-indigo-600 animate-pulse" : "bg-gray-700"}`}>
                    {done ? "✓" : active ? "·" : ""}
                  </div>
                  <span className={`text-xs ${active ? "text-white font-semibold" : done ? "text-gray-400" : "text-gray-600"}`}>
                    {label}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Overall progress bar */}
          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${Math.round((stageIndex / stages.length) * 100)}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-2 text-right">
            {Math.round((stageIndex / stages.length) * 100)}%
          </p>
        </div>
      )}

      {/* Error state */}
      {phase === "error" && error && (
        <div className="w-full max-w-xl rounded-2xl border border-red-700 bg-red-950/30 p-6">
          <p className="text-red-400 font-semibold mb-2">Analysis failed</p>
          <p className="text-red-300 text-sm font-mono break-all">{error}</p>
          <button
            onClick={() => { setPhase("idle"); setError(null); }}
            className="mt-4 px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white text-sm transition-colors"
          >
            Try again
          </button>
        </div>
      )}

      {/* Pipeline info */}
      <div className="mt-8 grid grid-cols-3 gap-4 w-full max-w-xl">
        {[
          { icon: "🔍", label: "18-stage pipeline", desc: "Scene → detect → track → motion → validate" },
          { icon: "🔁", label: "Closed-loop", desc: "Render + SSIM compare + refinement" },
          { icon: "📋", label: "Structured JSON", desc: "Renderable, reusable, confidence-scored" },
        ].map(({ icon, label, desc }) => (
          <div key={label} className="rounded-xl bg-gray-900 border border-gray-700 p-3 text-center">
            <div className="text-xl mb-1">{icon}</div>
            <p className="text-xs font-semibold text-gray-300 mb-0.5">{label}</p>
            <p className="text-xs text-gray-500">{desc}</p>
          </div>
        ))}
      </div>
    </main>
  );
}
