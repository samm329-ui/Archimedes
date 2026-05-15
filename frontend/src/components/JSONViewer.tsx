"use client";

interface JSONViewerProps {
  data: object | string;
  title?: string;
}

export default function JSONViewer({ data, title }: JSONViewerProps) {
  const jsonStr = typeof data === "string" ? data : JSON.stringify(data, null, 2);

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      {title && <h2 className="text-lg font-semibold text-white mb-4">{title}</h2>}
      <pre className="bg-slate-900 p-4 rounded-lg text-xs text-slate-300 overflow-auto max-h-96">
        {jsonStr}
      </pre>
    </div>
  );
}