import React, { useEffect, useRef, useState } from "react";

export default function App() {
  const [files, setFiles] = useState([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState(null);
  const [results, setResults] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [pageState, setPageState] = useState("upload");
  const [displayProgress, setDisplayProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const completeTimerRef = useRef(null);
  const pathname = window.location.pathname;
  const routePage = pathname === "/loading" ? "loading" : pathname === "/result" ? "result" : "upload";
  const currentPage = routePage === "upload" ? pageState : routePage;

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
    setErrorMsg("");
    setPageState("loading");
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("max_len", "64600");
    const res = await fetch("/api/predict_batch", { method: "POST", body: fd });
    const j = await parseApiResponse(res);
    if (!res.ok) {
      setPageState("upload");
      setErrorMsg(j.detail || "提交失败");
      return;
    }
    setJobId(j.job_id);
    setJob({ status: "pending", done_files: 0, total_files: files.length });
    setResults([]);
    setDisplayProgress(5);
  }

  useEffect(() => {
    if (pageState !== "loading" || !jobId) return undefined;

    let cancelled = false;
    const totalFiles = job?.total_files || files.length || 1;
    const maxPolls = Math.max(300, totalFiles * 240);

    async function pollJob() {
      const res = await fetch(`/api/tasks/${jobId}`);
      const j = await parseApiResponse(res);
      if (cancelled) return "stop";

      if (!res.ok) {
        setPageState("upload");
        setErrorMsg(j.detail || "查询任务失败");
        return "stop";
      }

      setJob(j);
      const total = j.total_files || totalFiles;
      const done = j.done_files || 0;
      const ratioProgress = total > 0 ? Math.round((done / total) * 100) : 0;

      if (j.status === "completed") {
        setDisplayProgress(100);
        setResults(j.items || []);
        completeTimerRef.current = setTimeout(() => {
          if (!cancelled) setPageState("result");
        }, 600);
        return "stop";
      }

      if (j.status === "failed") {
        setPageState("upload");
        setErrorMsg(j.error || "检测失败");
        return "stop";
      }

      const inFlight = (j.status === "running" || j.status === "pending") && done < total;
      const target = inFlight
        ? Math.max(8, ratioProgress, Math.min(94, Math.round(((done + 0.4) / total) * 100)))
        : Math.max(8, ratioProgress);
      setDisplayProgress((prev) => Math.max(prev, target));
      return "continue";
    }

    async function pollLoop() {
      for (let i = 0; i < maxPolls; i += 1) {
        const outcome = await pollJob();
        if (cancelled || outcome === "stop") return;
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
      if (!cancelled) {
        setPageState("upload");
        setErrorMsg("检测超时，请重试");
      }
    }

    pollLoop();

    return () => {
      cancelled = true;
      if (completeTimerRef.current) clearTimeout(completeTimerRef.current);
    };
  }, [pageState, jobId, files.length]);

  useEffect(() => {
    if (pageState !== "loading" || !job) return undefined;
    if (job.status === "completed" || job.status === "failed") return undefined;

    const total = job.total_files || files.length || 1;
    const done = job.done_files || 0;
    if (done >= total) return undefined;

    const timer = setInterval(() => {
      setDisplayProgress((prev) => {
        const ceiling = Math.min(94, Math.round(((done + 1) / total) * 100) - 1);
        if (prev >= ceiling) return prev;
        return Math.min(ceiling, prev + 0.6);
      });
    }, 180);

    return () => clearInterval(timer);
  }, [pageState, job, files.length]);

  function resetToUpload() {
    setPageState("upload");
    setJobId("");
    setJob(null);
    setResults([]);
    setErrorMsg("");
    setFiles([]);
    setDisplayProgress(0);
  }

  function updateFiles(fileList) {
    const picked = Array.from(fileList || []);
    setFiles(picked);
    setErrorMsg("");
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    updateFiles(e.dataTransfer.files);
  }

  function handleDragOver(e) {
    e.preventDefault();
    setIsDragOver(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setIsDragOver(false);
  }

  if (currentPage === "loading") {
    const done = job?.done_files || 0;
    const total = job?.total_files || files.length || 1;
    const progress = Math.max(0, Math.min(100, displayProgress));
    const activeIndex = Math.min(done + 1, total);
    const currentName = job?.current_filename || files[activeIndex - 1]?.name || "";
    const statusText = done >= total
      ? "检测完成，正在生成结果..."
      : currentName
        ? `正在检测：${currentName}`
        : "系统正在进行音频取证分析，请稍候...";

    return (
      <div className="forensicPage">
        <div className="brandCorner">
          <div className="brandDot">♪</div>
          <div>
            <p className="brandMain">音频伪造检测器</p>
            <p className="brandSub">CHECK</p>
          </div>
        </div>
        <div className="forensicCard">
          <div className="forensicWave">| | | | |</div>
          <h2>正在检测</h2>
          <p className="loadingStatusText">{statusText}</p>
          <div className="progressTrack">
            <div className="progressFill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progressText">{done}/{total} 已完成 · {Math.round(progress)}%</p>
        </div>
      </div>
    );
  }

  if (currentPage === "result") {
    const displayResults = results;
    const spoofCount = displayResults.filter((item) => item.decision_by_threshold === "spoof").length;
    const bonaCount = displayResults.length - spoofCount;
    const isBatch = displayResults.length > 1;
    const first = displayResults[0] || {};
    const headlineIsBona = isBatch ? bonaCount >= spoofCount : first.decision_by_threshold !== "spoof";
    const headlineScoreRaw = headlineIsBona ? first.prob_bonafide : first.prob_spoof;
    const headlineScore = !isBatch && Number.isFinite(headlineScoreRaw) ? Math.round(headlineScoreRaw * 100) : null;

    return (
      <div className="container">
        <header className="hero">
          <h1>音频伪造检测器</h1>
          <p className="heroSub">上传音频文件，辨别其是由AI生成还是真人录制。</p>
          <p className="heroMeta">保护自己免受语音克隆诈骗和深度伪造音频的侵害。</p>
        </header>
        <section className="card resultHeroCard">
          <div className="resultPosterTop">
            <p className="posterBrand">{isBatch ? "批量检测结果" : "检测结果"}</p>
            {isBatch && <span className="posterTag">共 {displayResults.length} 个文件</span>}
          </div>
          {!isBatch && (
            <div className="resultWave" aria-hidden="true">
              {Array.from({ length: 52 }).map((_, idx) => (
                <span key={idx} style={{ height: `${28 + ((idx * 17) % 52)}px` }} />
              ))}
            </div>
          )}
          <div className="resultHeadline">
            {isBatch ? (
              <p className="headlineValue batchSummaryValue">
                真实 {bonaCount} / 伪造 {spoofCount}
              </p>
            ) : (
              <p className="headlineValue">{headlineScore !== null ? `${headlineScore}%` : "RESULT"}</p>
            )}
            <span className={headlineIsBona ? "resultBadge bonaBadge" : "resultBadge spoofBadge"}>
              {isBatch
                ? headlineIsBona
                  ? "整体偏真实"
                  : "整体偏伪造"
                : headlineIsBona
                  ? "真实语音"
                  : "伪造语音"}
            </span>
          </div>
          <p className="headlineDesc">
            {isBatch
              ? `已完成 ${displayResults.length} 个文件的检测，详细结果见下方列表。`
              : headlineIsBona
                ? "该段音频更接近真人录制特征。"
                : "该段音频更接近伪造/合成语音特征。"}
          </p>
          {isBatch && (
            <p className="headlineMeta">
              统计：真实 {bonaCount} 条，伪造 {spoofCount} 条
            </p>
          )}
          <div className="row resultActionRow">
            <button onClick={resetToUpload}>继续检测</button>
          </div>
        </section>

        {isBatch && (
          <section className="card batchResultsCard">
            <h2>各文件检测详情</h2>
            <div className="batchTableWrap">
              <table className="batchResultsTable">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>文件名</th>
                    <th>判定结果</th>
                    <th>真实概率</th>
                    <th>伪造概率</th>
                    <th>耗时</th>
                  </tr>
                </thead>
                <tbody>
                  {displayResults.map((item, idx) => {
                    const isSpoof = item.decision_by_threshold === "spoof";
                    return (
                      <tr key={`${item.filename}-${idx}`}>
                        <td>{idx + 1}</td>
                        <td className="filenameCell" title={item.filename}>{item.filename}</td>
                        <td>
                          <span className={`badge ${isSpoof ? "spoofBadge" : "bonaBadge"}`}>
                            {isSpoof ? "伪造" : "真实"}
                          </span>
                        </td>
                        <td>{Number.isFinite(item.prob_bonafide) ? `${(item.prob_bonafide * 100).toFixed(1)}%` : "-"}</td>
                        <td>{Number.isFinite(item.prob_spoof) ? `${(item.prob_spoof * 100).toFixed(1)}%` : "-"}</td>
                        <td>{Number.isFinite(item.inference_time_ms) ? `${Math.round(item.inference_time_ms)} ms` : "-"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="container">
      <header className="hero">
        <h1>音频伪造检测器</h1>
        <p className="heroSub">上传音频文件，辨别其是由AI生成还是真人录制。</p>
        <p className="heroMeta">保护自己免受语音克隆诈骗和深度伪造音频的侵害。</p>
      </header>

      <section className="layoutGrid">
        <div className="panelCard uploadCard">
          <div
            className={`dropArea dragDropArea ${isDragOver ? "dragOver" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="dropIcon">◉</div>
            <h2>将音频文件拖放到此处</h2>
            <p className="muted">或点击下方按钮选择文件</p>
            <p className="fastHint">快速获取检测结果</p>
            <input
              ref={fileInputRef}
              className="hiddenFileInput"
              style={{ display: "none" }}
              type="file"
              multiple
              accept="audio/*"
              onChange={(e) => updateFiles(e.target.files)}
            />
            {!!files.length && (
              <p className="pickedFiles">已选择 {files.length} 个文件：{files.map((f) => f.name).join("，")}</p>
            )}
            <div className="row centerRow">
              <button
                type="button"
                className="pickBtn"
                onClick={files.length ? submitBatch : () => fileInputRef.current?.click()}
              >
                {files.length ? "开始检测" : "选择要分析的音频"}
              </button>
            </div>
            <p className="muted smallMuted">
              允许扩展名：.wav / .flac / .ogg / .opus / .mp3 / .aac / .m4a / .mp4
            </p>
            {errorMsg && <p className="error">{errorMsg}</p>}
          </div>
        </div>
      </section>
    </div>
  );
}
