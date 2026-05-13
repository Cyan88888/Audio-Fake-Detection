import React, { useRef, useState } from "react";

export default function App() {
  const [files, setFiles] = useState([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState(null);
  const [results, setResults] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [pageState, setPageState] = useState("upload");
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
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
    startAutoPoll(j.job_id);
  }

  function resetToUpload() {
    setPageState("upload");
    setJobId("");
    setJob(null);
    setResults([]);
    setErrorMsg("");
    setFiles([]);
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

  function startAutoPoll(targetJobId) {
    let count = 0;
    const timer = setInterval(async () => {
      count += 1;
      const res = await fetch(`/api/tasks/${targetJobId}`);
      const j = await parseApiResponse(res);
      if (res.ok) {
        setJob(j);
        if (j.status === "completed") {
          const items = j.items || [];
          setResults(items);
          setPageState("result");
          clearInterval(timer);
        } else if (j.status === "failed") {
          setPageState("upload");
          setErrorMsg(j.error || "检测失败");
          clearInterval(timer);
        }
      } else {
        setPageState("upload");
        setErrorMsg(j.detail || "查询任务失败");
        clearInterval(timer);
      }
      if (count >= 60) {
        setPageState("upload");
        setErrorMsg("检测超时，请重试");
        clearInterval(timer);
      }
    }, 1000);
  }

  if (currentPage === "loading") {
    const done = job?.done_files || 0;
    const total = job?.total_files || files.length || 1;
    const progress = Math.max(8, Math.min(96, (done / total) * 100));
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
          <p>系统正在进行音频取证分析，请稍候...</p>
          <div className="progressTrack">
            <div className="progressFill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progressText">{done}/{total} 已完成</p>
        </div>
      </div>
    );
  }

  if (currentPage === "result") {
    const displayResults = results.length
      ? results
      : [
          { filename: "sample_01.wav", decision_by_threshold: "bonafide" },
          { filename: "sample_02.wav", decision_by_threshold: "spoof" },
        ];
    const spoofCount = displayResults.filter((item) => item.decision_by_threshold === "spoof").length;
    const bonaCount = displayResults.length - spoofCount;
    const headlineIsBona = bonaCount >= spoofCount;
    const first = displayResults[0] || {};
    const headlineScoreRaw = headlineIsBona ? first.prob_bonafide : first.prob_spoof;
    const headlineScore = Number.isFinite(headlineScoreRaw) ? Math.round(headlineScoreRaw * 100) : null;
    return (
      <div className="container">
        <header className="hero">
          <h1>音频伪造检测器</h1>
          <p className="heroSub">上传音频文件，辨别其是由AI生成还是真人录制。</p>
          <p className="heroMeta">保护自己免受语音克隆诈骗和深度伪造音频的侵害。</p>
        </header>
        <section className="card resultHeroCard">
          <div className="resultPosterTop">
            <p className="posterBrand">检测结果</p>
          </div>
          <div className="resultWave" aria-hidden="true">
            {Array.from({ length: 52 }).map((_, idx) => (
              <span key={idx} style={{ height: `${28 + ((idx * 17) % 52)}px` }} />
            ))}
          </div>
          <div className="resultHeadline">
            <p className="headlineValue">{headlineScore !== null ? `${headlineScore}%` : "RESULT"}</p>
            <span className={headlineIsBona ? "resultBadge bonaBadge" : "resultBadge spoofBadge"}>
              {headlineIsBona ? "真实语音" : "伪造语音"}
            </span>
          </div>
          <p className="headlineDesc">
            {headlineIsBona
              ? "该段音频更接近真人录制特征。"
              : "该段音频更接近伪造/合成语音特征。"}
          </p>
          {/* <p className="headlineMeta">记录统计：真实 {bonaCount} 条，伪造 {spoofCount} 条</p> */}
          <div className="row resultActionRow">
            <button onClick={resetToUpload}>继续检测</button>
          </div>
        </section>
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
            <p className="fastHint">3秒获取结果</p>
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
