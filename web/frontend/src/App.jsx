import React, { useMemo, useState } from "react";

function WavePlot({ data }) {
  if (!data?.length) return null;
  const points = data.slice(0, 200).map((v, i) => `${i},${40 - v * 30}`).join(" ");
  return (
    <svg viewBox="0 0 200 80" className="plot">
      <polyline points={points} fill="none" stroke="#4ade80" strokeWidth="1.5" />
    </svg>
  );
}

function MelHeatmap({ mel }) {
  if (!mel?.length || !mel[0]?.length) return null;
  const rows = mel.length;
  const cols = mel[0].length;
  const min = Math.min(...mel.flat());
  const max = Math.max(...mel.flat());
  const cellW = 320 / cols;
  const cellH = 128 / rows;
  return (
    <svg viewBox="0 0 320 128" className="plot">
      {mel.map((row, r) =>
        row.map((v, c) => {
          const t = (v - min) / (max - min + 1e-8);
          const color = `rgb(${Math.floor(t * 255)}, ${Math.floor(t * 160)}, ${Math.floor(t * 80)})`;
          return <rect key={`${r}-${c}`} x={c * cellW} y={r * cellH} width={cellW + 0.2} height={cellH + 0.2} fill={color} />;
        })
      )}
    </svg>
  );
}

export default function App() {
  const [token] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [files, setFiles] = useState([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState(null);
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);

  const authHeader = useMemo(
    () => (token ? { Authorization: `Bearer ${token}` } : {}),
    [token]
  );

  async function parseApiResponse(res) {
    const text = await res.text();
    if (!text) return {};
    try {
      return JSON.parse(text);
    } catch (_e) {
      return { detail: text };
    }
  }

  async function submitBatch() {
    if (!files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("threshold", String(threshold));
    fd.append("max_len", "64600");
    const res = await fetch("/api/predict_batch", { method: "POST", headers: authHeader, body: fd });
    const j = await parseApiResponse(res);
    if (!res.ok) {
      alert(j.detail || "提交失败");
      return;
    }
    setJobId(j.job_id);
    setJob({ status: "pending", done_files: 0, total_files: files.length });
    setResults([]);
    setSelected(null);
    startAutoPoll(j.job_id);
  }

  async function pollTask() {
    if (!jobId) return;
    const res = await fetch(`/api/tasks/${jobId}`, { headers: authHeader });
    const j = await parseApiResponse(res);
    if (!res.ok) {
      alert(j.detail || "查询任务失败");
      return;
    }
    setJob(j);
    if (j.status === "completed") {
      const items = j.items || [];
      setResults(items);
      setSelected(items[0] ?? null);
    }
  }

  function startAutoPoll(targetJobId) {
    let count = 0;
    const timer = setInterval(async () => {
      count += 1;
      const res = await fetch(`/api/tasks/${targetJobId}`, { headers: authHeader });
      const j = await parseApiResponse(res);
      if (res.ok) {
        setJob(j);
        if (j.status === "completed") {
          const items = j.items || [];
          setResults(items);
          setSelected(items[0] ?? null);
          clearInterval(timer);
        } else if (j.status === "failed") {
          clearInterval(timer);
        }
      }
      if (count >= 60) clearInterval(timer);
    }, 1000);
  }

  return (
    <div className="container">
      <h1>S2 音频伪造检测系统</h1>
      <p className="muted">检测页面：批量文件检测与可视化结果展示</p>

      <section className="card">
        <h2>文件检测</h2>
        <label>阈值: {threshold.toFixed(2)}</label>
        <input type="range" min="0.1" max="0.9" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
        <input type="file" multiple accept="audio/*" onChange={(e) => setFiles(Array.from(e.target.files || []))} />
        <div className="row">
          <button onClick={submitBatch} disabled={!files.length}>开始检测</button>
          <button onClick={pollTask} disabled={!jobId}>刷新状态</button>
        </div>
        {job && <p className="hint">任务状态：{job.status}（{job.done_files || 0}/{job.total_files || 0}）</p>}
        {job?.status === "failed" && (
          <p className="error">任务失败：{job?.error || "未知错误"}</p>
        )}
      </section>

      <section className="card">
        <h2>检测结果</h2>
        {results.length === 0 && <p className="hint">暂无结果，请先选择音频并点击“开始检测”。</p>}
        <table>
          <thead>
            <tr>
              <th>文件</th>
              <th>P(spoof)</th>
              <th>P(bonafide)</th>
              <th>状态</th>
              <th>耗时(ms)</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {results.map((item, idx) => (
              <tr key={`${item.filename}-${idx}`}>
                <td>{item.filename}</td>
                <td>{(item.prob_spoof * 100).toFixed(2)}%</td>
                <td>{(item.prob_bonafide * 100).toFixed(2)}%</td>
                <td>
                  <span className={item.decision_by_threshold === "spoof" ? "badge spoofBadge" : "badge bonaBadge"}>
                    {item.decision_by_threshold === "spoof" ? "伪造" : "真实"}
                  </span>
                </td>
                <td>{item.inference_time_ms?.toFixed(1)}</td>
                <td><button onClick={() => setSelected(item)}>查看</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {selected && (
        <section className="card">
          <h2>可视化详情</h2>
          <p>
            {selected.filename} | 判定: <b>{selected.decision_by_threshold}</b> | P(spoof): {(selected.prob_spoof * 100).toFixed(2)}%
          </p>
          <WavePlot data={selected.waveform} />
          <MelHeatmap mel={selected.mel_db} />
        </section>
      )}
    </div>
  );
}
