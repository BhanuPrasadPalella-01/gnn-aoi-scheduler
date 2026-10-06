// Thin wrapper around the local Python server (gnn_aoi/server.py).

async function call(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const api = {
  datasets: () => call("/api/datasets"),
  results: () => call("/api/results"),
  session: (opts) => call("/api/session", opts),
  step: (id, n = 1) => call("/api/step", { id, n }),
};
