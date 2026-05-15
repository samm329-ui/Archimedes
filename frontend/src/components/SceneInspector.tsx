"use client";

interface Scene {
  id: string;
  start_frame: number;
  end_frame: number;
  confidence: number;
  elements_count?: number;
}

interface SceneInspectorProps {
  scenes: Scene[];
  selected?: string;
  onSelect: (id: string) => void;
}

export default function SceneInspector({ scenes, selected, onSelect }: SceneInspectorProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-white mb-4">Scene Inspector</h2>
      <div className="space-y-2 max-h-96 overflow-auto">
        {scenes.map((scene) => (
          <div
            key={scene.id}
            className={`p-3 rounded-lg cursor-pointer transition-colors ${
              selected === scene.id ? "bg-purple-600" : "bg-slate-700 hover:bg-slate-600"
            }`}
            onClick={() => onSelect(scene.id)}
          >
            <div className="flex justify-between items-start">
              <div>
                <div className="font-medium text-white">Scene {scene.id}</div>
                <div className="text-sm text-slate-400">
                  Frames {scene.start_frame} - {scene.end_frame}
                </div>
              </div>
              <div className="text-sm font-medium text-purple-400">
                {Math.round(scene.confidence * 100)}%
              </div>
            </div>
            {scene.elements_count !== undefined && (
              <div className="mt-1 text-xs text-slate-400">
                {scene.elements_count} elements
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}