"use client";

import { useState } from "react";

interface FrameViewerProps {
  frames: Array<{
    index: number;
    imageUrl?: string;
    elements?: Array<{
      id: string;
      bbox: { x: number; y: number; w: number; h: number };
    }>;
  }>;
  currentFrame: number;
  onFrameChange: (frame: number) => void;
}

export default function FrameViewer({ frames = [], currentFrame = 0, onFrameChange }: FrameViewerProps) {
  const frame = frames[currentFrame];

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Frame Preview</h2>
      <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-video">
        <div className="absolute inset-0 flex items-center justify-center text-slate-500">
          Frame {frames.length > 0 ? `${currentFrame + 1} / ${frames.length}` : "No frames"}
        </div>
        {frame?.elements && frame.elements.length > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {frame.elements.map((el) => (
              <g key={el.id}>
                <rect
                  x={el.bbox.x}
                  y={el.bbox.y}
                  width={el.bbox.w}
                  height={el.bbox.h}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                />
              </g>
            ))}
          </svg>
        )}
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, frames.length - 1)}
        value={currentFrame}
        onChange={(e) => onFrameChange(Number(e.target.value))}
        className="w-full mt-4"
      />
    </div>
  );
}