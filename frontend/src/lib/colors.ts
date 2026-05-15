export function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "text-emerald-400";
  if (confidence >= 0.6) return "text-amber-400";
  return "text-red-400";
}

export function confidenceBg(confidence: number): string {
  if (confidence >= 0.8) return "bg-emerald-500";
  if (confidence >= 0.6) return "bg-amber-500";
  return "bg-red-500";
}

export function elementColor(index: number): string {
  const colors = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
    "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
  ];
  return colors[index % colors.length];
}

export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export function hexToRgb(hex: string): [number, number, number] | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : null;
}