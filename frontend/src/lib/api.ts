const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function analyzeVideo(file: File): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`);
  }

  return response.json();
}

export async function analyzeVideoPath(path: string): Promise<any> {
  const response = await fetch(`${API_BASE}/analyze/path?path=${encodeURIComponent(path)}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`);
  }

  return response.json();
}

export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_BASE}/health`);
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}