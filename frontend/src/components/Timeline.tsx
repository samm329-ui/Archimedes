"use client";

interface TimelineTrack {
  id: string;
  color: string;
  segments: Array<{
    start_frame: number;
    end_frame: number;
  }>;
}

interface TimelineProps {
  totalFrames: number;
  currentFrame: number;
  onFrameChange: (frame: number) => void;
  tracks?: TimelineTrack[];
}

export default function Timeline({ totalFrames, currentFrame, onFrameChange, tracks }: TimelineProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Timeline</h2>
      <div className="relative h-24 bg-slate-900 rounded-lg overflow-hidden">
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
          style={{ left: `${(currentFrame / totalFrames) * 100}%` }}
        />
        {tracks && tracks.map((track) => (
          <div key={track.id} className="absolute left-0 right-0 h-6" style={{ top: 0 }}>
            {track.segments.map((seg, i) => (
              <div
                key={i}
                className="absolute h-full opacity-50"
                style={{
                  backgroundColor: track.color,
                  left: `${(seg.start_frame / totalFrames) * 100}%`,
                  width: `${((seg.end_frame - seg.start_frame) / totalFrames) * 100}%`,
                }}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-xs text-slate-400">
        <span>Frame 0</span>
        <span className="font-mono text-white">{currentFrame} / {totalFrames}</span>
        <span>Frame {totalFrames}</span>
      </div>
      <input
        type="range"
        min={0}
        max={totalFrames}
        value={currentFrame}
        onChange={(e) => onFrameChange(Number(e.target.value))}
        className="w-full mt-2"
      />
    </div>
  );
}