/* Trauma Behaviour Video Annotator — frontend */

const BEHAVIOURS = {
  flashback: { label: "Flashback", color: "#a855f7" },
  avoidance: { label: "Avoidance", color: "#f59e0b" },
  negative_emotion: { label: "Negative emotion", color: "#3b82f6" },
  hyper_arousal: { label: "Hyperarousal", color: "#f97316" },
  normal: { label: "Normal", color: "#22c55e" },
};
const POI_COLOR = "#ef233c";
const ID_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16", "#14b8a6", "#f97316", "#6366f1"];
const IMAGE_CACHE_MAX = 90;

const state = {
  config: null,
  videos: [],
  filter: "all",
  search: "",
  collapsedGroups: new Set(JSON.parse(localStorage.getItem("collapsedGroups") || "[]")),
  vid: null,
  video: null,
  frames: [],
  ann: {},
  idx: 0,
  videoLabel: null,
  hasUndo: false,
  image: null,
  imageCache: new Map(),
  loadToken: 0,
  selectedId: null,
  drag: null,
  zoom: 1,
  scale: 1,
  playing: false,
  idNums: new Map(),
  nextNum: 1,
  jobs: [],
  seenFinished: new Set(),
  waiters: new Map(),
  overlayJob: null,
  pollTimer: null,
  lastVideoRefresh: 0,
  pendingImport: null,
  trackOpts: JSON.parse(localStorage.getItem("trackOpts") || "null"),
};

const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------- utilities

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.json || opts.body ? "POST" : "GET") };
  if (opts.json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  } else if (opts.body) {
    init.body = opts.body;
  }
  const res = await fetch(path, init);
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = body?.detail ?? body;
    const msg = typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail);
    const err = new Error(msg || `HTTP ${res.status}`);
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return body;
}

function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), kind === "error" ? 6000 : 3500);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function uid(prefix = "obj") {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

function download(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function dialogOpen() {
  return !!document.querySelector("dialog[open]") || !$("wait-overlay").classList.contains("hidden");
}

// ---------------------------------------------------------------- wait overlay

function showWait(message, job = null) {
  state.overlayJob = job?.id || null;
  $("wait-message").textContent = message;
  $("wait-sub").textContent = job?.message || "";
  $("wait-bar").style.width = `${Math.round((job?.progress || 0) * 100)}%`;
  $("wait-overlay").querySelector(".progress").classList.toggle("hidden", !job);
  $("wait-actions").classList.toggle("hidden", !job);
  $("wait-overlay").classList.remove("hidden");
}

function hideWait() {
  state.overlayJob = null;
  $("wait-overlay").classList.add("hidden");
}

async function withWait(message, fn) {
  showWait(message);
  try {
    return await fn();
  } finally {
    hideWait();
  }
}

// ---------------------------------------------------------------- saving

const dirty = new Map(); // frame name -> { vid, slot, fields:Set }
let saveTimer = null;
let saveChain = Promise.resolve();
let metaTimer = null;

function setSaveStatus(kind) {
  const el = $("save-status");
  el.className = `pill ${kind}`;
  el.textContent = { saving: "Saving…", saved: "All changes saved", error: "Save failed — retrying" }[kind] || kind;
}

function markDirty(name, ...fields) {
  const slot = state.ann[name];
  if (!slot || !state.vid) return;
  let e = dirty.get(name);
  if (!e || e.vid !== state.vid) {
    e = { vid: state.vid, slot, fields: new Set() };
    dirty.set(name, e);
  }
  fields.forEach((f) => e.fields.add(f));
  setSaveStatus("saving");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSaves, 400);
  scheduleTimeline();
}

function flushSaves() {
  clearTimeout(saveTimer);
  const run = saveChain.then(async () => {
    const entries = [...dirty.entries()];
    dirty.clear();
    let failed = false;
    for (const [name, e] of entries) {
      const body = { frame: name };
      if (e.fields.has("behaviours")) body.behaviours = e.slot.behaviours;
      if (e.fields.has("comment")) body.comment = e.slot.comment;
      if (e.fields.has("objects")) body.objects = e.slot.objects.map(cleanObject);
      try {
        await api(`/api/annotate/${enc(e.vid)}`, { json: body });
      } catch (err) {
        if (err.status && err.status < 500) {
          toast(`Could not save ${name}: ${err.message}`, "error");
          continue;
        }
        failed = true;
        const again = dirty.get(name) || { vid: e.vid, slot: e.slot, fields: new Set() };
        e.fields.forEach((f) => again.fields.add(f));
        dirty.set(name, again);
      }
    }
    if (failed) {
      setSaveStatus("error");
      saveTimer = setTimeout(flushSaves, 3000);
    } else if (!dirty.size) {
      setSaveStatus("saved");
    }
  });
  saveChain = run.catch(() => {});
  return run;
}

function cleanObject(o) {
  return {
    id: String(o.id),
    bbox: o.bbox.map((v) => Math.round(v * 100) / 100),
    label: o.is_poi ? "person_of_interest" : o.label === "person_of_interest" ? "person" : o.label || "person",
    behaviours: o.behaviours || [],
    confirmed: o.confirmed !== false,
    source: o.source || "manual",
    conf: o.conf ?? null,
    is_poi: !!o.is_poi,
    poi_locked: !!(o.is_poi && o.poi_locked),
  };
}

function scheduleMetaSave() {
  setSaveStatus("saving");
  clearTimeout(metaTimer);
  metaTimer = setTimeout(saveMeta, 500);
}

async function saveMeta() {
  clearTimeout(metaTimer);
  metaTimer = null;
  if (!state.vid) return;
  const vid = state.vid;
  try {
    await api(`/api/video-meta/${enc(vid)}`, {
      json: {
        video_label: state.videoLabel || "",
        video_comment: $("video-comment").value,
        context: $("context").value,
        participant_id: $("participant-id").value,
        session_id: $("session-id").value,
      },
    });
    if (!dirty.size) setSaveStatus("saved");
    const v = state.videos.find((x) => x.id === vid);
    if (v) {
      v.video_label = state.videoLabel;
      renderLibrary();
    }
  } catch (e) {
    setSaveStatus("error");
    toast(`Could not save video info: ${e.message}`, "error");
  }
}

async function flushAll() {
  if (metaTimer) await saveMeta();
  await flushSaves();
}

window.addEventListener("beforeunload", (e) => {
  if (dirty.size || metaTimer) {
    flushAll();
    e.preventDefault();
    e.returnValue = "";
  }
});

// ---------------------------------------------------------------- init

async function init() {
  try {
    state.config = await api("/api/config");
  } catch (e) {
    toast(`Server not reachable: ${e.message}`, "error");
    return;
  }
  $("data-dir").textContent = state.config.data_dir;
  restoreDetectSettings();
  buildBehaviourButtons();
  buildLegend();
  bindUi();
  await refreshVideos();
  await pollJobs(true);
  if (!state.config.yolo_ready) toast("YOLO (ultralytics) is not installed — detection and tracking are disabled", "error");
}

// v2: default model changed nano -> small; older saved settings lose their model once.
const DETECT_SETTINGS_VERSION = 2;

function restoreDetectSettings() {
  const defaults = state.config.detect_defaults || {};
  let saved = JSON.parse(localStorage.getItem("detectSettings") || "null");
  if (saved && (saved.settingsVersion || 1) < DETECT_SETTINGS_VERSION) {
    delete saved.model;
    saved.settingsVersion = DETECT_SETTINGS_VERSION;
    localStorage.setItem("detectSettings", JSON.stringify(saved));
  }
  saved = saved || defaults;
  const model = saved.model || defaults.model;
  if (model) $("detect-model").value = model;
  if (saved.imgsz) $("detect-imgsz").value = String(saved.imgsz);
  if (saved.conf) $("detect-conf").value = saved.conf;
  $("detect-augment").checked = !!saved.augment;
  $("detect-lying").checked = saved.lying !== false;
  $("detect-conf-val").textContent = Number($("detect-conf").value).toFixed(2);
}

function detectSettings() {
  const s = {
    model: $("detect-model").value,
    imgsz: +$("detect-imgsz").value,
    conf: +$("detect-conf").value,
    augment: $("detect-augment").checked,
    lying: $("detect-lying").checked,
  };
  localStorage.setItem("detectSettings", JSON.stringify({ ...s, settingsVersion: DETECT_SETTINGS_VERSION }));
  return s;
}

function buildBehaviourButtons() {
  const wrap = $("behaviour-btns");
  wrap.innerHTML = "";
  (state.config.behaviours || Object.keys(BEHAVIOURS)).forEach((b, i) => {
    const meta = BEHAVIOURS[b] || { label: b, color: "#888" };
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "beh";
    btn.dataset.behaviour = b;
    btn.style.setProperty("--c", meta.color);
    btn.innerHTML = `<kbd>${i + 1}</kbd><span class="dot"></span>${escapeHtml(meta.label)}`;
    btn.onclick = () => toggleBehaviour(b);
    wrap.appendChild(btn);
  });
}

function buildLegend() {
  $("legend").innerHTML = Object.entries(BEHAVIOURS)
    .map(([, m]) => `<span><i style="background:${m.color}"></i>${escapeHtml(m.label)}</span>`)
    .join("");
}

function bindUi() {
  $("btn-open-folder").onclick = openFolderModal;
  $("btn-open-folder-2").onclick = openFolderModal;
  $("btn-upload-video").onclick = () => $("file-video").click();
  $("btn-upload-frames").onclick = () => $("file-frames").click();
  $("btn-import-json").onclick = () => {
    if (!state.vid) return toast("Open a video first, then import its JSON", "error");
    $("file-json").click();
  };
  $("file-video").onchange = onVideoPicked;
  $("file-frames").onchange = onFramesPicked;
  $("file-json").onchange = onJsonPicked;

  $("btn-collapse").onclick = () => setSidebar(false);
  $("btn-expand").onclick = () => setSidebar(true);

  $("lib-search").oninput = (e) => { state.search = e.target.value.toLowerCase(); renderLibrary(); };
  document.querySelectorAll("#lib-filter button").forEach((b) => {
    b.onclick = () => {
      state.filter = b.dataset.filter;
      document.querySelectorAll("#lib-filter button").forEach((x) => x.classList.toggle("active", x === b));
      renderLibrary();
    };
  });

  $("btn-prev-video").onclick = () => stepVideo(-1);
  $("btn-next-video").onclick = () => stepVideo(1);
  $("btn-help").onclick = () => $("help-modal").showModal();

  $("btn-first").onclick = () => go(0);
  $("btn-prev").onclick = (e) => go(state.idx - (e.shiftKey ? 10 : 1));
  $("btn-next").onclick = (e) => go(state.idx + (e.shiftKey ? 10 : 1));
  $("btn-last").onclick = () => go(state.frames.length - 1);
  $("btn-play").onclick = togglePlay;
  $("frame-input").onchange = (e) => go((+e.target.value || 1) - 1);
  $("btn-zoom-in").onclick = () => setZoom(state.zoom * 1.25);
  $("btn-zoom-out").onclick = () => setZoom(state.zoom / 1.25);
  $("btn-zoom-fit").onclick = () => setZoom(1);

  document.querySelectorAll("#video-label-btns .vlabel").forEach((btn) => {
    btn.onclick = () => {
      state.videoLabel = state.videoLabel === btn.dataset.vlabel ? null : btn.dataset.vlabel;
      syncVideoLabel();
      saveMeta();
    };
  });
  ["participant-id", "session-id", "video-comment", "context"].forEach((id) => {
    $(id).addEventListener("input", scheduleMetaSave);
  });
  $("frame-comment").addEventListener("input", (e) => {
    const name = state.frames[state.idx];
    if (!name) return;
    state.ann[name].comment = e.target.value;
    markDirty(name, "comment");
  });

  $("detect-conf").oninput = (e) => { $("detect-conf-val").textContent = Number(e.target.value).toFixed(2); };
  $("btn-detect").onclick = runDetect;
  $("btn-detect-track").onclick = () => runDetectTrack();
  $("btn-detect-track-opts").onclick = openDetectTrackModal;
  $("dt-summary").onclick = openDetectTrackModal;
  syncDetectTrackSummary();
  $("btn-track").onclick = openTrackModal;
  $("btn-undo").onclick = undo;
  $("btn-fill-next").onclick = fillUntilNext;
  $("btn-fill-n").onclick = () => fillRange(state.idx + 1, state.idx + Math.max(1, +$("fill-n").value || 1));

  $("btn-export-json").onclick = () => exportVideo("json");
  $("btn-export-coco").onclick = () => exportVideo("coco");
  $("btn-export-all").onclick = exportAll;

  $("folder-form").onsubmit = onFolderSubmit;
  $("btn-browse").onclick = browseFolder;
  $("track-form").onsubmit = onTrackSubmit;
  $("dt-form").onsubmit = onDetectTrackSubmit;
  $("conflict-form").onsubmit = onConflictSubmit;
  $("wipe-form").onsubmit = onWipeSubmit;
  $("wipe-confirm").oninput = () => {
    $("wipe-go").disabled = $("wipe-confirm").value !== state.pendingImport?.probe?.video_id;
  };
  $("wait-bg").onclick = () => { hideWait(); toast("Running in the background — see progress in the sidebar", "ok"); };
  $("wait-cancel").onclick = () => state.overlayJob && cancelJob(state.overlayJob);

  const c = $("canvas");
  c.addEventListener("mousedown", onCanvasDown);
  window.addEventListener("mousemove", onCanvasMove);
  window.addEventListener("mouseup", onCanvasUp);
  c.addEventListener("mousemove", onCanvasHover);
  c.addEventListener("dblclick", onCanvasDblClick);
  $("canvas-wrap").addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    setZoom(state.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
  }, { passive: false });

  const tl = $("timeline");
  tl.addEventListener("mousedown", (e) => {
    const seek = (ev) => {
      const r = tl.getBoundingClientRect();
      go(Math.floor(clamp((ev.clientX - r.left) / r.width, 0, 0.9999) * state.frames.length));
    };
    seek(e);
    const move = (ev) => seek(ev);
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });

  bindBoxTable();
  window.addEventListener("resize", () => { fitCanvas(); drawTimeline(); });
  window.addEventListener("keydown", onKey);
}

function setSidebar(show) {
  $("app").classList.toggle("sidebar-collapsed", !show);
  $("btn-expand").classList.toggle("hidden", show);
  requestAnimationFrame(() => { fitCanvas(); drawTimeline(); });
}

// ---------------------------------------------------------------- library

async function refreshVideos() {
  try {
    const data = await api("/api/videos");
    state.videos = data.videos || [];
    state.lastVideoRefresh = Date.now();
  } catch (e) {
    toast(`Could not load video list: ${e.message}`, "error");
  }
  renderLibrary();
}

function visibleVideos() {
  return state.videos.filter((v) => {
    if (state.filter === "todo" && v.video_label) return false;
    if (state.filter === "done" && !v.video_label) return false;
    if (state.search) {
      const hay = `${v.name} ${v.id} ${v.group}`.toLowerCase();
      if (!hay.includes(state.search)) return false;
    }
    return true;
  });
}

function renderLibrary() {
  const list = $("lib-list");
  const vids = visibleVideos();
  $("lib-empty").classList.toggle("hidden", state.videos.length > 0);
  const groups = new Map();
  vids.forEach((v) => {
    if (!groups.has(v.group)) groups.set(v.group, []);
    groups.get(v.group).push(v);
  });
  const html = [];
  for (const [group, items] of groups) {
    const collapsed = state.collapsedGroups.has(group);
    const done = items.filter((v) => v.video_label).length;
    html.push(`<div class="group${collapsed ? " collapsed" : ""}">
      <button type="button" class="group-head" data-group="${escapeHtml(group)}" title="${escapeHtml(group)}">
        <span class="caret">${collapsed ? "▸" : "▾"}</span><span class="gname">${escapeHtml(group)}</span>
        <span class="gcount">${done}/${items.length}</span></button>
      <div class="group-items">`);
    items.forEach((v) => {
      const pct = v.num_frames ? Math.round((100 * v.labelled) / v.num_frames) : 0;
      const statusText = {
        queued: "waiting to extract…", extracting: "extracting frames…", error: `error: ${v.error || ""}`,
        cancelled: "not extracted (cancelled)", empty: "no frames",
      }[v.status] || `${v.num_frames} frames · ${pct}% labelled · ${v.boxed} boxed`;
      const lab = v.video_label === "trauma" ? `<span class="vi-label trauma">Trauma</span>`
        : v.video_label === "no_trauma" ? `<span class="vi-label no-trauma">No trauma</span>` : "";
      html.push(`<button type="button" class="vid-item ${v.status}${v.id === state.vid ? " active" : ""}" data-vid="${escapeHtml(v.id)}" title="${escapeHtml(v.name)}">
        <span class="vi-status"></span>
        <span class="vi-main"><span class="vi-name">${escapeHtml(v.name)}</span>
          <span class="vi-meta">${escapeHtml(statusText)}</span>
          ${v.num_frames ? `<span class="vi-bar"><i style="width:${pct}%"></i></span>` : ""}</span>
        ${lab}</button>`);
    });
    html.push(`</div></div>`);
  }
  list.innerHTML = html.join("");
  list.querySelectorAll(".group-head").forEach((b) => {
    b.onclick = () => {
      const g = b.dataset.group;
      if (state.collapsedGroups.has(g)) state.collapsedGroups.delete(g);
      else state.collapsedGroups.add(g);
      localStorage.setItem("collapsedGroups", JSON.stringify([...state.collapsedGroups]));
      renderLibrary();
    };
  });
  list.querySelectorAll(".vid-item").forEach((b) => {
    b.onclick = () => openVideo(b.dataset.vid);
  });
  const total = state.videos.length;
  const labelled = state.videos.filter((v) => v.video_label).length;
  const ready = state.videos.filter((v) => v.num_frames).length;
  $("lib-summary").textContent = total ? `${total} videos · ${ready} ready · ${labelled} labelled Trauma / No trauma` : "";
}

function stepVideo(dir) {
  const ready = visibleVideos().filter((v) => v.num_frames);
  if (!ready.length) return;
  const i = ready.findIndex((v) => v.id === state.vid);
  const next = ready[clamp(i + dir, 0, ready.length - 1)];
  if (next && next.id !== state.vid) openVideo(next.id);
}

// ---------------------------------------------------------------- open video / frames

async function openVideo(vid) {
  const v = state.videos.find((x) => x.id === vid);
  if (v && (!v.num_frames || v.status === "queued" || v.status === "extracting")) {
    toast(v.status === "error" ? `Extraction failed: ${v.error}` : "This video is not extracted yet — it will be ready soon", v.status === "error" ? "error" : "");
    return;
  }
  stopPlay();
  await flushAll();
  let data;
  try {
    data = await api(`/api/video/${enc(vid)}`);
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  state.vid = data.video_id;
  state.video = data;
  state.frames = data.frames || [];
  state.ann = data.annotations || {};
  state.videoLabel = data.video_label;
  state.hasUndo = !!data.has_undo;
  state.idx = 0;
  state.zoom = 1;
  state.selectedId = null;
  state.image = null;
  state.imageCache.clear();
  rebuildIdNums();

  $("empty-state").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  $("video-title").textContent = data.name;
  $("video-title").title = data.source_path || data.name;
  $("video-sub").textContent = `${data.group} · ${state.frames.length} frames${data.fps ? ` · ${data.fps} fps` : ""} · id ${data.video_id}`;
  $("frame-total").textContent = state.frames.length;
  $("frame-input").max = state.frames.length;
  $("participant-id").value = data.participant_id || "";
  $("session-id").value = data.session_id || "";
  $("video-comment").value = data.video_comment || "";
  $("context").value = data.context || "";
  $("export-result").classList.add("hidden");
  syncVideoLabel();
  syncUndo(data.undo_action);
  setSaveStatus("saved");
  renderLibrary();
  document.querySelector(`.vid-item[data-vid="${CSS.escape(state.vid)}"]`)?.scrollIntoView({ block: "nearest" });
  await go(0);
}

async function reloadAnnotations() {
  if (!state.vid) return;
  await flushSaves();
  const data = await api(`/api/video/${enc(state.vid)}`);
  if ((data.frames || []).length !== state.frames.length || data.frames_version !== state.video?.frames_version) {
    const idx = state.idx;
    await openVideo(state.vid);
    await go(idx);
    return;
  }
  state.ann = data.annotations || {};
  state.hasUndo = !!data.has_undo;
  if (!metaTimer) {
    state.videoLabel = data.video_label;
    $("participant-id").value = data.participant_id || "";
    $("session-id").value = data.session_id || "";
    syncVideoLabel();
  }
  syncUndo(data.undo_action);
  rebuildIdNums();
  renderFrameUi();
  draw();
  drawTimeline();
}

function curName() {
  return state.frames[state.idx];
}

function curSlot() {
  const name = curName();
  if (!name) return { behaviours: [], comment: "", bbox: null, objects: [] };
  if (!state.ann[name]) state.ann[name] = { behaviours: [], comment: "", bbox: null, objects: [] };
  const s = state.ann[name];
  s.behaviours = s.behaviours || [];
  s.objects = s.objects || [];
  return s;
}

function frameUrl(name) {
  return `/api/frame/${enc(state.vid)}/${enc(name)}?v=${state.video?.frames_version || 0}`;
}

function loadImage(name) {
  const key = `${state.vid}/${name}`;
  let entry = state.imageCache.get(key);
  if (!entry) {
    const img = new Image();
    img.decoding = "async";
    const promise = new Promise((resolve, reject) => {
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Cannot load ${name}`));
    });
    img.src = frameUrl(name);
    entry = { img, promise };
    state.imageCache.set(key, entry);
    if (state.imageCache.size > IMAGE_CACHE_MAX) {
      state.imageCache.delete(state.imageCache.keys().next().value);
    }
  }
  return entry.promise;
}

async function go(idx) {
  if (!state.frames.length) return;
  idx = clamp(Math.round(idx), 0, state.frames.length - 1);
  const prevSel = state.selectedId;
  state.idx = idx;
  state.drag = null;
  const slot = curSlot();
  state.selectedId = slot.objects.some((o) => o.id === prevSel) ? prevSel : slot.objects.find((o) => o.is_poi)?.id || null;
  renderFrameUi();
  drawTimeline();
  const token = ++state.loadToken;
  try {
    const img = await loadImage(state.frames[idx]);
    if (token !== state.loadToken) return;
    const sizeChanged = !state.image || state.image.naturalWidth !== img.naturalWidth || state.image.naturalHeight !== img.naturalHeight;
    state.image = img;
    if (sizeChanged) fitCanvas();
    else draw();
  } catch (e) {
    if (token === state.loadToken) toast(e.message, "error");
  }
  for (let k = 1; k <= 4; k++) {
    if (idx + k < state.frames.length) loadImage(state.frames[idx + k]).catch(() => {});
  }
}

function renderFrameUi() {
  const slot = curSlot();
  const name = curName();
  $("frame-input").value = state.idx + 1;
  $("frame-name").textContent = name || "";
  const fps = state.video?.fps;
  $("frame-time").textContent = fps ? fmtTime(state.idx / fps) : "";
  if (document.activeElement !== $("frame-comment")) $("frame-comment").value = slot.comment || "";
  syncBehaviourButtons();
  renderPeople();
}

function syncVideoLabel() {
  document.querySelectorAll("#video-label-btns .vlabel").forEach((b) => {
    b.classList.toggle("active", b.dataset.vlabel === state.videoLabel);
  });
}

function syncUndo(action) {
  $("btn-undo").disabled = !state.hasUndo;
  $("btn-undo").textContent = state.hasUndo && action ? `Undo ${action.replace("_", "-")}` : "Undo";
}

// ---------------------------------------------------------------- objects / POI

function rebuildIdNums() {
  state.idNums = new Map();
  state.nextNum = 1;
  state.frames.forEach((n) => (state.ann[n]?.objects || []).forEach((o) => numOf(o)));
}

function numOf(o) {
  const id = String(o.id);
  if (!state.idNums.has(id)) state.idNums.set(id, state.nextNum++);
  return state.idNums.get(id);
}

function colorOf(o) {
  if (o.is_poi) return POI_COLOR;
  return ID_COLORS[(numOf(o) - 1) % ID_COLORS.length];
}

function selectedObject() {
  return curSlot().objects.find((o) => o.id === state.selectedId) || null;
}

// locked = the user picked this POI, so Detect / Auto-VIP will not replace it with the largest person.
function setPoi(id, locked = true) {
  const slot = curSlot();
  const target = slot.objects.find((o) => o.id === id);
  if (!target) return;
  slot.objects.forEach((o) => {
    o.is_poi = o === target;
    o.poi_locked = o.is_poi && locked;
    if (o.is_poi) {
      o.label = "person_of_interest";
      o.confirmed = true;
    } else if (o.label === "person_of_interest") {
      o.label = "person";
    }
  });
  slot.bbox = [...target.bbox];
  // Mirrors server sync_poi: the POI carries the frame behaviours (adopting its own if the frame has none).
  if (!slot.behaviours.length && target.behaviours?.length) slot.behaviours = cleanBehaviours(target.behaviours);
  target.behaviours = [...slot.behaviours];
  state.selectedId = id;
  markDirty(curName(), "objects");
  syncBehaviourButtons();
  renderPeople();
  draw();
}

function deleteObject(id) {
  const slot = curSlot();
  const victim = slot.objects.find((o) => o.id === id);
  if (!victim) return;
  slot.objects = slot.objects.filter((o) => o !== victim);
  if (victim.is_poi) slot.bbox = null;
  if (state.selectedId === id) state.selectedId = slot.objects.find((o) => o.is_poi)?.id || null;
  markDirty(curName(), "objects");
  renderPeople();
  draw();
}

function clearObjects() {
  const slot = curSlot();
  if (!slot.objects.length) return;
  slot.objects = [];
  slot.bbox = null;
  state.selectedId = null;
  markDirty(curName(), "objects");
  renderPeople();
  draw();
}

const OBJECT_LABELS = { person_of_interest: "Person of interest", person: "Person", other: "Other" };
const BEH_SHORT = { flashback: "Flash", avoidance: "Avoid", negative_emotion: "Neg. emo", hyper_arousal: "Hyper", normal: "Normal" };

function behaviourKeys() {
  return state.config?.behaviours || Object.keys(BEHAVIOURS);
}

function cleanBehaviours(list) {
  const keys = behaviourKeys();
  const out = [...new Set((list || []).filter((b) => keys.includes(b)))];
  return out.includes("normal") && out.length > 1 ? out.filter((b) => b !== "normal") : out;
}

function sourceText(o) {
  if (isUserBox(o)) return "manual";
  if (o.source === "manual" || o.source === "legacy") return "old annotation";
  return { auto_vip: "auto", detection: "detected", tracker: "tracked", visual: "tracked", hold: "tracked", seed: "tracked", lying: "tracked (lying)" }[o.source] || o.source;
}

function metaText(o) {
  return [o.conf != null ? `${Math.round(o.conf * 100)}%` : "", sourceText(o)].filter(Boolean).join(" · ");
}

let lastRenderedSel = null;

function renderPeople() {
  const slot = curSlot();
  const objs = [...slot.objects].sort((a, b) => (b.is_poi - a.is_poi) || (numOf(a) - numOf(b)));
  const n = objs.length;
  const poi = objs.find((o) => o.is_poi);
  $("people-summary").innerHTML = n
    ? `${n} box${n > 1 ? "es" : ""} on this frame · ${poi ? `POI is <strong>ID ${numOf(poi)}</strong>` : "<strong>no POI</strong>"}. Edit them in the table under the image.`
    : "No boxes on this frame. Draw one on the image, press <kbd>D</kbd>, or <kbd>Shift+D</kbd> for all frames.";
  $("boxes-count").textContent = n ? `(${n})` : "";
  $("boxes-table").classList.toggle("hidden", !n);
  $("boxes-empty").classList.toggle("hidden", n > 0);
  $("btn-clear-objects").classList.toggle("hidden", n < 2);
  const W = state.image?.naturalWidth || "";
  const H = state.image?.naturalHeight || "";
  $("boxes-body").innerHTML = objs.map((o) => {
    const active = new Set(o.is_poi ? slot.behaviours : o.behaviours || []);
    const chips = behaviourKeys().map((b) => {
      const meta = BEHAVIOURS[b] || { label: b, color: "#888" };
      const tip = o.is_poi ? `${meta.label} — frame behaviour (same as the Behaviour buttons / keys 1–5)` : `${meta.label} — this box only`;
      return `<button type="button" class="chip${active.has(b) ? " on" : ""}" data-b="${escapeHtml(b)}" style="--c:${meta.color}" title="${escapeHtml(tip)}">${escapeHtml(BEH_SHORT[b] || meta.label)}</button>`;
    }).join("");
    const label = o.is_poi ? "person_of_interest" : o.label === "other" ? "other" : "person";
    const options = Object.entries(OBJECT_LABELS)
      .map(([v, t]) => `<option value="${v}"${v === label ? " selected" : ""}>${t}</option>`).join("");
    const coords = ["x1", "y1", "x2", "y2"].map((name, i) =>
      `<input type="number" class="b-coord" data-i="${i}" step="1" min="0" max="${i % 2 ? H : W}" value="${Math.round(o.bbox[i])}" title="${name}" aria-label="${name}" />`).join("");
    return `<tr class="box-row${o.is_poi ? " poi" : ""}${o.id === state.selectedId ? " selected" : ""}" data-id="${escapeHtml(o.id)}">
      <td class="c-id"><span class="swatch" style="background:${colorOf(o)}"></span>ID ${numOf(o)}</td>
      <td>${o.is_poi ? `<span class="tag-poi">POI</span>` : `<button type="button" class="mk-poi" title="Make this the person of interest (P)">Make POI</button>`}</td>
      <td><select class="b-label" title="Label">${options}</select></td>
      <td class="c-beh">${chips}</td>
      <td class="c-box">${coords}</td>
      <td class="c-meta muted small">${escapeHtml(metaText(o))}</td>
      <td><button type="button" class="del icon ghost" title="Delete box (Del)">✕</button></td>
    </tr>`;
  }).join("");
  if (state.selectedId && state.selectedId !== lastRenderedSel) scrollRowIntoView(state.selectedId);
  lastRenderedSel = state.selectedId;
}

function boxRow(id) {
  return [...$("boxes-body").rows].find((r) => r.dataset.id === id) || null;
}

function scrollRowIntoView(id) {
  const row = boxRow(id);
  const sc = $("boxes-scroll");
  if (!row) return;
  const head = $("boxes-table").tHead.offsetHeight;
  const top = row.offsetTop - head;
  const bottom = row.offsetTop + row.offsetHeight;
  if (top < sc.scrollTop) sc.scrollTop = top;
  else if (bottom > sc.scrollTop + sc.clientHeight) sc.scrollTop = bottom - sc.clientHeight;
}

function selectRow(id) {
  if (state.selectedId === id) return;
  state.selectedId = id;
  lastRenderedSel = id;
  [...$("boxes-body").rows].forEach((r) => r.classList.toggle("selected", r.dataset.id === id));
  draw();
}

function setObjectLabel(id, label) {
  const slot = curSlot();
  const o = slot.objects.find((x) => x.id === id);
  if (!o) return;
  if (label === "person_of_interest") return setPoi(id);
  if (o.is_poi) {
    o.is_poi = false;
    o.poi_locked = false;
    slot.bbox = null;
  }
  o.label = label;
  markDirty(curName(), "objects");
  renderPeople();
  draw();
}

function toggleObjectBehaviour(id, b) {
  const o = curSlot().objects.find((x) => x.id === id);
  if (!o) return;
  if (o.is_poi) return toggleBehaviour(b);
  let list = cleanBehaviours(o.behaviours);
  if (b === "normal") list = list.includes("normal") ? [] : ["normal"];
  else {
    list = list.filter((x) => x !== "normal");
    list = list.includes(b) ? list.filter((x) => x !== b) : [...list, b];
  }
  o.behaviours = list;
  markDirty(curName(), "objects");
  renderPeople();
}

// Live on "input" (only while the box stays valid); on "change" the value is clamped and x1<x2, y1<y2 enforced.
function editCoord(input, commit) {
  const row = input.closest("tr");
  const slot = curSlot();
  const o = slot.objects.find((x) => x.id === row?.dataset.id);
  if (!o || !state.image) return;
  const W = state.image.naturalWidth;
  const H = state.image.naturalHeight;
  const i = +input.dataset.i;
  const v = parseFloat(input.value);
  const inputs = row.querySelectorAll(".b-coord");
  const writeBack = () => inputs.forEach((el, j) => { el.value = Math.round(o.bbox[j]); el.classList.remove("bad"); });
  if (!isFinite(v)) {
    if (commit) writeBack();
    return;
  }
  let bb = [...o.bbox];
  bb[i] = clamp(v, 0, i % 2 ? H : W);
  if (commit) bb = normBox(...bb);
  const valid = bb[2] - bb[0] >= 1 && bb[3] - bb[1] >= 1;
  input.classList.toggle("bad", !valid);
  if (!valid) {
    if (commit) writeBack();
    return;
  }
  if (bb.some((x, j) => x !== o.bbox[j])) {
    o.bbox = bb;
    o.source = "manual";
    o.confirmed = true;
    if (o.is_poi) slot.bbox = [...bb];
    markDirty(curName(), "objects");
    row.querySelector(".c-meta").textContent = metaText(o);
    draw();
  }
  if (commit) writeBack();
}

function bindBoxTable() {
  const body = $("boxes-body");
  body.addEventListener("click", (e) => {
    const row = e.target.closest("tr.box-row");
    if (!row) return;
    const id = row.dataset.id;
    const btn = e.target.closest("button");
    if (btn?.classList.contains("del")) return deleteObject(id);
    selectRow(id);
    if (btn?.classList.contains("mk-poi")) setPoi(id);
    else if (btn?.classList.contains("chip")) toggleObjectBehaviour(id, btn.dataset.b);
  });
  body.addEventListener("focusin", (e) => {
    const row = e.target.closest("tr.box-row");
    if (row) selectRow(row.dataset.id);
  });
  body.addEventListener("input", (e) => {
    if (e.target.classList.contains("b-coord")) editCoord(e.target, false);
  });
  body.addEventListener("change", (e) => {
    if (e.target.classList.contains("b-coord")) editCoord(e.target, true);
    else if (e.target.classList.contains("b-label")) setObjectLabel(e.target.closest("tr").dataset.id, e.target.value);
  });
  $("btn-clear-objects").onclick = clearObjects;

  const panel = $("boxes-panel");
  const scroller = $("boxes-scroll");
  panel.open = localStorage.getItem("boxesOpen") !== "0";
  const savedH = +localStorage.getItem("boxesHeight");
  if (savedH) scroller.style.height = `${savedH}px`;
  panel.addEventListener("toggle", () => localStorage.setItem("boxesOpen", panel.open ? "1" : "0"));

  let queued = false;
  const refit = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fitCanvas(); });
  };
  new ResizeObserver(() => {
    if (panel.open && scroller.offsetHeight) localStorage.setItem("boxesHeight", String(scroller.offsetHeight));
    refit();
  }).observe(panel);
  new ResizeObserver(refit).observe(document.querySelector(".stage"));
}

// Server-converted legacy boxes get "obj_" ids; boxes drawn, moved or resized in this UI keep other ids.
function isUserBox(o) {
  return o.source === "manual" && !String(o.id).startsWith("obj_");
}

// Saved data has no poi_locked flag: automatic sources count as unlocked, hand-drawn/moved boxes as locked.
function isPoiLocked(o) {
  return !!o?.is_poi && (!!o.poi_locked || isUserBox(o));
}

function boxArea(o) {
  return (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]);
}

// The automatic POI is the largest visible person (ties: higher confidence).
function smartPoi(slot, candidates = slot.objects) {
  let best = null;
  candidates.forEach((o) => {
    if (!best || boxArea(o) > boxArea(best) || (boxArea(o) === boxArea(best) && (o.conf ?? 0) > (best.conf ?? 0))) best = o;
  });
  if (best) setPoi(best.id, false);
  return best;
}

// ---------------------------------------------------------------- canvas

function fitCanvas() {
  const img = state.image;
  const canvas = $("canvas");
  const wrap = $("canvas-wrap");
  if (!img) return;
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const fit = Math.min((wrap.clientWidth - 16) / W, (wrap.clientHeight - 16) / H);
  state.scale = Math.max(0.05, fit * state.zoom);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${Math.round(W * state.scale)}px`;
  canvas.style.height = `${Math.round(H * state.scale)}px`;
  canvas.width = Math.round(W * state.scale * dpr);
  canvas.height = Math.round(H * state.scale * dpr);
  draw();
}

function setZoom(z) {
  state.zoom = clamp(z, 0.3, 6);
  fitCanvas();
}

function draw() {
  const canvas = $("canvas");
  const img = state.image;
  if (!img) return;
  const ctx = canvas.getContext("2d");
  const k = canvas.width / img.naturalWidth;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(k, 0, 0, k, 0, 0);
  ctx.drawImage(img, 0, 0);
  const px = 1 / k;
  const slot = curSlot();
  const drag = state.drag;
  const objs = [...slot.objects].sort((a, b) => a.is_poi - b.is_poi);
  objs.forEach((o) => {
    let bb = o.bbox;
    if (drag && drag.id === o.id && drag.box) bb = drag.box;
    const [x1, y1, x2, y2] = bb;
    const color = colorOf(o);
    const sel = o.id === state.selectedId;
    ctx.lineWidth = (o.is_poi ? 3 : 2) * px * (sel ? 1.4 : 1);
    ctx.strokeStyle = color;
    ctx.setLineDash(o.confirmed === false ? [6 * px, 4 * px] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (o.is_poi) {
      ctx.fillStyle = "rgba(239,35,60,0.10)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    const tag = `${o.is_poi ? "POI · " : ""}ID ${numOf(o)}${o.conf != null ? ` ${Math.round(o.conf * 100)}%` : ""}`;
    ctx.font = `600 ${12 * px}px Segoe UI, sans-serif`;
    const tw = ctx.measureText(tag).width + 8 * px;
    const th = 16 * px;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, x1 + 4 * px, ty + 12 * px);
    if (sel) {
      const hs = 7 * px;
      [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([hx, hy]) => {
        ctx.fillStyle = "#fff";
        ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
        ctx.strokeStyle = color;
        ctx.lineWidth = px;
        ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
      });
    }
  });
  if (drag && drag.mode === "draw" && drag.box) {
    const [x1, y1, x2, y2] = drag.box;
    ctx.strokeStyle = slot.objects.some((o) => o.is_poi) ? "#ffffff" : POI_COLOR;
    ctx.lineWidth = 2 * px;
    ctx.setLineDash([6 * px, 4 * px]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }
}

function imgPoint(e) {
  const r = $("canvas").getBoundingClientRect();
  const img = state.image;
  const s = r.width / (img?.naturalWidth || 1);
  return { x: (e.clientX - r.left) / s, y: (e.clientY - r.top) / s, s };
}

function normBox(x1, y1, x2, y2) {
  const W = state.image.naturalWidth;
  const H = state.image.naturalHeight;
  return [clamp(Math.min(x1, x2), 0, W), clamp(Math.min(y1, y2), 0, H), clamp(Math.max(x1, x2), 0, W), clamp(Math.max(y1, y2), 0, H)];
}

function hitHandle(o, p) {
  if (!o) return null;
  const tol = 9 / p.s;
  const [x1, y1, x2, y2] = o.bbox;
  const pts = { nw: [x1, y1], ne: [x2, y1], sw: [x1, y2], se: [x2, y2] };
  for (const [name, [hx, hy]] of Object.entries(pts)) {
    if (Math.abs(p.x - hx) <= tol && Math.abs(p.y - hy) <= tol) return name;
  }
  return null;
}

function hitObject(p) {
  const inside = curSlot().objects.filter((o) => p.x >= o.bbox[0] && p.x <= o.bbox[2] && p.y >= o.bbox[1] && p.y <= o.bbox[3]);
  if (!inside.length) return null;
  const sel = inside.find((o) => o.id === state.selectedId);
  if (sel) return sel;
  return inside.sort((a, b) => (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]) - (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))[0];
}

function onCanvasDown(e) {
  if (!state.image || e.button !== 0) return;
  e.preventDefault();
  stopPlay();
  const p = imgPoint(e);
  const sel = selectedObject();
  const handle = hitHandle(sel, p);
  if (handle) {
    state.drag = { mode: "resize", id: sel.id, handle, start: p, orig: [...sel.bbox], box: [...sel.bbox] };
    return;
  }
  const hit = hitObject(p);
  if (hit) {
    state.selectedId = hit.id;
    state.drag = { mode: "move", id: hit.id, start: p, orig: [...hit.bbox], box: null };
    renderPeople();
    draw();
    return;
  }
  state.drag = { mode: "draw", start: p, box: [p.x, p.y, p.x, p.y] };
}

function onCanvasMove(e) {
  const d = state.drag;
  if (!d || !state.image) return;
  const p = imgPoint(e);
  if (d.mode === "draw") {
    d.box = normBox(d.start.x, d.start.y, p.x, p.y);
  } else if (d.mode === "move") {
    const dx = p.x - d.start.x;
    const dy = p.y - d.start.y;
    if (!d.box && Math.hypot(dx, dy) * p.s < 3) return;
    const [x1, y1, x2, y2] = d.orig;
    const W = state.image.naturalWidth;
    const H = state.image.naturalHeight;
    const cx = clamp(dx, -x1, W - x2);
    const cy = clamp(dy, -y1, H - y2);
    d.box = [x1 + cx, y1 + cy, x2 + cx, y2 + cy];
  } else if (d.mode === "resize") {
    let [x1, y1, x2, y2] = d.orig;
    if (d.handle.includes("w")) x1 = p.x;
    if (d.handle.includes("e")) x2 = p.x;
    if (d.handle.includes("n")) y1 = p.y;
    if (d.handle.includes("s")) y2 = p.y;
    d.box = normBox(x1, y1, x2, y2);
  }
  draw();
}

function onCanvasUp() {
  const d = state.drag;
  if (!d) return;
  state.drag = null;
  const slot = curSlot();
  if (d.mode === "draw") {
    const [x1, y1, x2, y2] = d.box;
    if (x2 - x1 >= 6 && y2 - y1 >= 6) {
      const makePoi = !slot.objects.some((o) => o.is_poi);
      const obj = {
        id: uid("man"), bbox: d.box, label: makePoi ? "person_of_interest" : "person", behaviours: [],
        confirmed: true, source: "manual", conf: null, is_poi: makePoi, poi_locked: makePoi,
      };
      slot.objects.push(obj);
      if (makePoi) slot.bbox = [...obj.bbox];
      state.selectedId = obj.id;
      markDirty(curName(), "objects");
      renderPeople();
    } else {
      state.selectedId = null;
      renderPeople();
    }
  } else if (d.box) {
    const o = slot.objects.find((x) => x.id === d.id);
    const [x1, y1, x2, y2] = d.box;
    if (o && x2 - x1 >= 4 && y2 - y1 >= 4) {
      o.bbox = d.box;
      o.source = "manual";
      o.confirmed = true;
      if (o.is_poi) slot.bbox = [...o.bbox];
      markDirty(curName(), "objects");
      renderPeople();
    }
  }
  draw();
}

function onCanvasHover(e) {
  if (state.drag || !state.image) return;
  const p = imgPoint(e);
  const h = hitHandle(selectedObject(), p);
  const c = $("canvas");
  if (h) c.style.cursor = h === "nw" || h === "se" ? "nwse-resize" : "nesw-resize";
  else if (hitObject(p)) c.style.cursor = "move";
  else c.style.cursor = "crosshair";
}

function onCanvasDblClick(e) {
  const hit = hitObject(imgPoint(e));
  if (hit && !hit.is_poi) {
    setPoi(hit.id);
    toast(`ID ${numOf(hit)} is now the person of interest`, "ok");
  }
}

// ---------------------------------------------------------------- timeline

let timelineQueued = false;
function scheduleTimeline() {
  if (timelineQueued) return;
  timelineQueued = true;
  requestAnimationFrame(() => { timelineQueued = false; drawTimeline(); });
}

function drawTimeline() {
  const c = $("timeline");
  const n = state.frames.length;
  const w = c.clientWidth;
  const h = c.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, w, h);
  if (!n) return;
  const bandH = h - 12;
  const cols = Math.min(n, Math.floor(w));
  for (let col = 0; col < cols; col++) {
    const a = Math.floor((col * n) / cols);
    const b = Math.max(a + 1, Math.floor(((col + 1) * n) / cols));
    let beh = null;
    let boxed = false;
    for (let i = a; i < b; i++) {
      const s = state.ann[state.frames[i]];
      if (!s) continue;
      if (s.bbox) boxed = true;
      const bs = s.behaviours || [];
      if (bs.length) {
        const pick = bs.find((x) => x !== "normal") || bs[0];
        if (!beh || beh === "normal") beh = pick;
      }
    }
    const x = (col * w) / cols;
    const cw = Math.max(1, w / cols + 0.5);
    if (beh) {
      ctx.fillStyle = BEHAVIOURS[beh]?.color || "#888";
      ctx.fillRect(x, 3, cw, bandH - 3);
    }
    if (boxed) {
      ctx.fillStyle = "#94a3b8";
      ctx.fillRect(x, h - 7, cw, 4);
    }
  }
  const cx = ((state.idx + 0.5) * w) / n;
  ctx.fillStyle = "#fff";
  ctx.fillRect(cx - 1, 0, 2, h);
}

// ---------------------------------------------------------------- behaviours

function syncBehaviourButtons() {
  const set = new Set(curSlot().behaviours);
  document.querySelectorAll("#behaviour-btns .beh").forEach((b) => b.classList.toggle("active", set.has(b.dataset.behaviour)));
}

function toggleBehaviour(b) {
  const name = curName();
  if (!name) return;
  const slot = curSlot();
  let list = [...slot.behaviours];
  if (b === "normal") list = list.includes("normal") ? [] : ["normal"];
  else {
    list = list.filter((x) => x !== "normal");
    list = list.includes(b) ? list.filter((x) => x !== b) : [...list, b];
  }
  slot.behaviours = list;
  const poi = slot.objects.find((o) => o.is_poi);
  if (poi) poi.behaviours = [...list];
  markDirty(name, "behaviours");
  syncBehaviourButtons();
  if (poi) renderPeople();
}

async function fillRange(start, end) {
  const beh = curSlot().behaviours;
  if (!beh.length) return toast("Select at least one behaviour on this frame first", "error");
  end = Math.min(end, state.frames.length - 1);
  if (start > end) return toast("No frames after this one", "error");
  await flushSaves();
  try {
    const res = await api(`/api/fill/${enc(state.vid)}`, { json: { start, end, behaviours: beh } });
    Object.assign(state.ann, res.annotations);
    state.hasUndo = true;
    syncUndo("fill");
    drawTimeline();
    toast(`Applied ${beh.map((b) => BEHAVIOURS[b]?.label || b).join(" + ")} to frames ${res.start + 1}–${res.end + 1}`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

function fillUntilNext() {
  let j = state.idx + 1;
  while (j < state.frames.length && !(state.ann[state.frames[j]]?.behaviours || []).length) j++;
  if (j === state.idx + 1) return toast("The next frame is already labelled", "error");
  fillRange(state.idx + 1, j - 1);
}

// ---------------------------------------------------------------- detection / VIP / tracking

async function runDetect() {
  if (!state.vid) return;
  const name = curName();
  const s = detectSettings();
  try {
    const res = await withWait("Detecting people…", () =>
      api(`/api/detect/${enc(state.vid)}/${enc(name)}`, { json: s }));
    if (curName() !== name) return;
    const slot = curSlot();
    // Each detection updates its best-matching automatic box in place (same ID, POI, label, behaviours);
    // user boxes are never changed. Weak (low_conf) matches don't overwrite existing boxes.
    const matched = new Set();
    const fresh = [];
    let refreshed = 0;
    [...res.detections].sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0)).forEach((d) => {
      let best = null;
      let bestIou = 0.5;
      slot.objects.forEach((o) => {
        const v = iou(o.bbox, d.bbox);
        if (v > bestIou && !matched.has(o)) { best = o; bestIou = v; }
      });
      if (!best) {
        if (!slot.objects.some((o) => iou(o.bbox, d.bbox) > 0.5)) fresh.push({ ...d });
        return;
      }
      matched.add(best);
      if (isUserBox(best) || res.low_conf) return;
      const same = best.source === d.source && best.conf === d.conf && best.bbox.every((v, i) => Math.abs(v - d.bbox[i]) < 0.5);
      if (same) return;
      best.bbox = [...d.bbox];
      best.conf = d.conf;
      best.source = d.source;
      if (best.is_poi) slot.bbox = [...best.bbox];
      refreshed++;
    });
    fresh.forEach((d) => slot.objects.push(d));
    if (fresh.length || refreshed) markDirty(name, "objects");
    const found = `${res.count} person(s) found · ${refreshed} box(es) refreshed · ${fresh.length} new.`;
    const oldPoi = slot.objects.find((o) => o.is_poi);
    const locked = isPoiLocked(oldPoi);
    const people = slot.objects.filter((o) => o.label !== "other" && (isUserBox(o) || matched.has(o) || fresh.includes(o)));
    const newPoi = !locked && people.length ? smartPoi(slot, people) : oldPoi;
    const moved = oldPoi && newPoi && newPoi !== oldPoi;
    const oldCovered = oldPoi && res.detections.some((d) => iou(d.bbox, oldPoi.bbox) > 0.3);
    renderPeople();
    draw();
    if (!res.count) toast(res.tip, "error");
    else if (res.low_conf) toast(res.tip);
    else if (moved) toast(`${found} The POI moved from ID ${numOf(oldPoi)} to ID ${numOf(newPoi)}, the largest person in the frame.${oldCovered ? "" : ` The old box covered no detected person — delete it with ✕ if it is wrong.`} Double-click a box (or Make POI) to choose the POI yourself.`);
    else if (locked) toast(`${found} Your chosen POI (ID ${numOf(oldPoi)}) was kept — double-click another box to change it.`);
    else toast(`${found} Red = person of interest (the largest person) — double-click another box to change.`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

function detectTrackOpts() {
  return { range: "all", poi: "largest", ...JSON.parse(localStorage.getItem("detectTrackOpts") || "null") };
}

function syncDetectTrackSummary() {
  const o = detectTrackOpts();
  $("dt-summary").textContent = `${o.range === "from" ? "From this frame to the end" : "Whole video"} · POI: ${
    o.poi === "follow" ? "follow the current red box" : "largest visible person"} · your own boxes kept · ⚙ options`;
}

function openDetectTrackModal() {
  const f = $("dt-form");
  const o = detectTrackOpts();
  f.range.value = o.range;
  f.poi.value = o.poi;
  $("dt-from").textContent = `frame ${state.idx + 1}`;
  const poi = state.vid ? curSlot().objects.find((x) => x.is_poi) : null;
  $("dt-follow-note").textContent = poi ? `(ID ${numOf(poi)} on frame ${state.idx + 1})` : "(no red box on this frame — double-click a person first)";
  const s = detectSettings();
  const model = $("detect-model").selectedOptions[0]?.textContent.split(" (")[0] || s.model;
  $("dt-settings").textContent = `Uses your Detection settings: model ${model} · lying-down search ${s.lying ? "on" : "off"}` +
    `${s.lying ? ` (lying-down boxes need ≥ ${s.conf.toFixed(2)} confidence)` : ""}. Tracking runs at 640 px.`;
  $("dt-modal").showModal();
}

async function onDetectTrackSubmit(e) {
  e.preventDefault();
  const action = e.submitter?.value;
  $("dt-modal").close();
  if (action !== "run" && action !== "save") return;
  const f = $("dt-form");
  const opts = { range: f.range.value, poi: f.poi.value };
  localStorage.setItem("detectTrackOpts", JSON.stringify(opts));
  syncDetectTrackSummary();
  if (action === "run") await runDetectTrack(opts);
}

async function runDetectTrack(opts = detectTrackOpts()) {
  if (!state.vid) return;
  const seedIdx = state.idx;
  const poi = curSlot().objects.find((o) => o.is_poi);
  const follow = opts.poi === "follow";
  if (follow && !poi) {
    toast("“Follow the current red POI” needs a red box on this frame: double-click the right person (or draw one), or choose “Largest visible person”.", "error");
    openDetectTrackModal();
    return;
  }
  await flushSaves();
  const s = detectSettings();
  const start = opts.range === "from" ? seedIdx : 0;
  try {
    const res = await api(`/api/detect-track/${enc(state.vid)}`, {
      json: {
        model: s.model, imgsz: 640, conf: s.conf, lying: s.lying, start, poi: opts.poi,
        seed_frame: follow ? seedIdx : null, seed_box: follow ? poi.bbox : null,
      },
    });
    kickJobs();
    const job = await waitForJob(res.job, `Detecting & tracking people in ${state.frames.length - start} frames`);
    if (job.status === "cancelled") return toast("Detect + track cancelled — nothing was changed");
    if (job.status !== "done") return;
    const span = job.result?.follow;
    if (span && span.stop_reason !== "end" && span.last + 1 < state.frames.length) {
      await go(span.last + 1);
      const why = span.stop_reason === "scene_cut" ? "scene cut" : "person lost";
      toast(`Detect + track finished: ${job.message}. Following stopped here (${why}); from this frame on the POI is as before. ` +
        "Double-click the person and press Shift+D with “From this frame” (⚙) to continue.");
    } else {
      toast(`Detect + track finished: ${job.message}`, "ok");
    }
  } catch (e) {
    toast(e.message, "error");
  }
}

function openTrackModal() {
  const f = $("track-form");
  const o = state.trackOpts;
  if (o) {
    ["propagate_bbox", "propagate_behaviours", "propagate_comment", "overwrite_bbox", "overwrite_labels", "stop_on_label_change"]
      .forEach((k) => { if (k in o) f[k].checked = !!o[k]; });
    f.range.value = o.num_frames > 0 ? "n" : "end";
    if (o.num_frames > 0) f.num_frames.value = o.num_frames;
    if (o.max_gap) f.max_gap.value = o.max_gap;
    if (o.iou_thresh) f.iou_thresh.value = o.iou_thresh;
  }
  $("track-modal").showModal();
}

function readTrackForm() {
  const f = $("track-form");
  return {
    propagate_bbox: f.propagate_bbox.checked,
    propagate_behaviours: f.propagate_behaviours.checked,
    propagate_comment: f.propagate_comment.checked,
    overwrite_bbox: f.overwrite_bbox.checked,
    overwrite_labels: f.overwrite_labels.checked,
    stop_on_label_change: f.stop_on_label_change.checked,
    num_frames: f.range.value === "end" ? 0 : Math.max(1, +f.num_frames.value || 60),
    max_gap: +f.max_gap.value || 15,
    iou_thresh: +f.iou_thresh.value || 0.3,
  };
}

async function onTrackSubmit(e) {
  e.preventDefault();
  const action = e.submitter?.value;
  $("track-modal").close();
  if (action !== "track") return;
  const opts = readTrackForm();
  state.trackOpts = opts;
  localStorage.setItem("trackOpts", JSON.stringify(opts));
  await runTrack(opts);
}

async function runTrack(opts) {
  if (!state.vid) return;
  const poi = curSlot().objects.find((o) => o.is_poi);
  if (opts.propagate_bbox && !poi) {
    toast("Mark the person of interest first: draw a box, or Detect and double-click the right person", "error");
    return;
  }
  await flushSaves();
  const s = detectSettings();
  try {
    const res = await api(`/api/track/${enc(state.vid)}/${state.idx}`, {
      json: { ...opts, seed_box: poi?.bbox || null, model: s.model, imgsz: 640 },
    });
    let result = res.result;
    if (res.job) {
      kickJobs();
      const job = await waitForJob(res.job, "Tracking the person of interest");
      if (job.status !== "done") return;
      result = job.result;
    } else {
      await reloadAnnotations();
    }
    const why = {
      end: "reached the end", lost: "person lost", scene_cut: "scene cut", label_change: "different label ahead", unreadable: "unreadable frame",
    }[result.stop_reason] || result.stop_reason;
    const stopped = result.stop_reason === "lost" || result.stop_reason === "scene_cut";
    toast(`Tracked ${result.covered} frame(s) — ${why}`, stopped ? "" : "ok");
    if (stopped && state.vid) {
      await go(Math.min(state.frames.length - 1, result.stopped_at + 1));
      toast(`Tracking stopped (${why}) after frame ${result.stopped_at + 1}. Mark the person on this frame and press T to continue.`, "");
    }
  } catch (e) {
    toast(e.message, "error");
  }
}

async function undo() {
  if (!state.vid || !state.hasUndo) return;
  await flushSaves();
  try {
    const res = await api(`/api/undo/${enc(state.vid)}`, { method: "POST" });
    await reloadAnnotations();
    toast(`Undid ${String(res.action || "last change").replace("_", "-")} on ${res.restored} frame(s)`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------------------------------------------------------------- jobs

function kickJobs() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(() => pollJobs(), 300);
}

function waitForJob(job, title) {
  return new Promise((resolve) => {
    state.waiters.set(job.id, { resolve, title });
    showWait(title, job);
  });
}

async function cancelJob(id) {
  try {
    await api(`/api/jobs/${id}/cancel`, { method: "POST" });
    toast("Cancelling…");
  } catch (e) {
    toast(e.message, "error");
  }
}

async function pollJobs(initial = false) {
  clearTimeout(state.pollTimer);
  let jobs = [];
  try {
    jobs = (await api("/api/jobs")).jobs || [];
  } catch {
    state.pollTimer = setTimeout(() => pollJobs(), 3000);
    return;
  }
  state.jobs = jobs;
  const finished = (j) => ["done", "error", "cancelled"].includes(j.status);
  let needVideos = false;
  let reloadCurrent = false;

  for (const j of jobs) {
    if (state.overlayJob === j.id) {
      $("wait-sub").textContent = j.message || "";
      $("wait-bar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
    }
    if (!finished(j) || state.seenFinished.has(j.id)) continue;
    state.seenFinished.add(j.id);
    if (initial) continue;
    needVideos = true;
    if (j.video_id && j.video_id === state.vid && j.kind !== "import") reloadCurrent = true;
    const waiter = state.waiters.get(j.id);
    if (j.status === "error") toast(`${j.title} failed: ${j.error}`, "error");
    else if (!waiter && j.status === "done") toast(`${j.title}: ${j.message}`, "ok");
    if (waiter) {
      state.waiters.delete(j.id);
      if (state.overlayJob === j.id) hideWait();
      if (reloadCurrent) { await reloadAnnotations(); reloadCurrent = false; }
      waiter.resolve(j);
    }
  }
  renderJobs();
  const active = jobs.some((j) => !finished(j));
  if (needVideos || (active && Date.now() - state.lastVideoRefresh > 2000)) {
    await refreshVideos();
    if (!state.vid) {
      const first = state.videos.find((v) => v.num_frames && v.status === "ready");
      if (first && jobs.some((j) => j.kind === "import")) openVideo(first.id);
    }
  }
  if (reloadCurrent) await reloadAnnotations();
  if (active || state.waiters.size) state.pollTimer = setTimeout(() => pollJobs(), 1000);
}

function renderJobs() {
  const panel = $("jobs-panel");
  const now = Date.now() / 1000;
  const shown = state.jobs.filter((j) => ["queued", "running"].includes(j.status) || (j.finished && now - j.finished < 8));
  panel.innerHTML = shown.map((j) => `
    <div class="job ${j.status}">
      <div class="job-top"><span class="job-title">${escapeHtml(j.title)}</span>
        ${["queued", "running"].includes(j.status) ? `<button type="button" class="icon ghost" data-cancel="${j.id}" title="Cancel">✕</button>` : `<span class="small">${j.status}</span>`}</div>
      <div class="progress"><div style="width:${Math.round((j.progress || 0) * 100)}%"></div></div>
      <div class="muted small job-msg">${escapeHtml(j.status === "queued" ? "Waiting in queue…" : j.message || "")}</div>
    </div>`).join("");
  panel.querySelectorAll("[data-cancel]").forEach((b) => { b.onclick = () => cancelJob(b.dataset.cancel); });
}

// ---------------------------------------------------------------- export

async function exportVideo(kind) {
  if (!state.vid) return;
  await flushAll();
  download(kind === "coco" ? `/api/export-coco/${enc(state.vid)}` : `/api/export/${enc(state.vid)}`);
  toast(`Downloading ${kind.toUpperCase()} — a copy is also saved in ${state.config.export_dir}`, "ok");
}

async function exportAll() {
  await flushAll();
  try {
    const res = await withWait("Exporting all videos…", () => api("/api/export-all", { method: "POST" }));
    const box = $("export-result");
    box.classList.remove("hidden");
    box.innerHTML = `<p><strong>${res.videos} video(s) exported</strong> to<br><span class="mono">${escapeHtml(res.folder)}</span></p>
      <div class="files">${res.files.map((f) => `<a class="button small" href="/api/exports/${enc(f)}" download>${escapeHtml(f)}</a>`).join("")}</div>
      <button type="button" class="link" id="btn-open-exports">Open export folder</button>`;
    $("btn-open-exports").onclick = () => api("/api/open-exports", { method: "POST" }).catch((e) => toast(e.message, "error"));
    toast("Export complete", "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---------------------------------------------------------------- import: folder

function openFolderModal() {
  const recent = JSON.parse(localStorage.getItem("recentFolders") || "[]");
  $("recent-folders").innerHTML = recent.length
    ? `<span class="muted small">Recent:</span> ${recent.map((p) => `<button type="button" class="link" data-path="${escapeHtml(p)}">${escapeHtml(p)}</button>`).join("")}`
    : "";
  $("recent-folders").querySelectorAll("[data-path]").forEach((b) => { b.onclick = () => { $("folder-path").value = b.dataset.path; }; });
  if (!$("folder-path").value && recent[0]) $("folder-path").value = recent[0];
  $("folder-modal").showModal();
  $("folder-path").focus();
}

async function browseFolder() {
  try {
    const res = await withWait("Choose a folder in the window that just opened (check the taskbar if you don't see it)…",
      () => api("/api/browse-folder"));
    if (res.path) $("folder-path").value = res.path;
  } catch (e) {
    toast(e.message, "error");
  }
}

async function onFolderSubmit(e) {
  e.preventDefault();
  if (e.submitter?.value !== "start") { $("folder-modal").close(); return; }
  const path = $("folder-path").value.trim().replace(/^"|"$/g, "");
  if (!path) return toast("Enter or browse to a folder first", "error");
  const fps = parseFloat($("folder-fps").value);
  try {
    const res = await api("/api/import/folder", {
      json: {
        path,
        fps: Number.isFinite(fps) && fps > 0 ? fps : null,
        label_from_folder: $("folder-labels").checked,
        auto_vip: $("folder-vip").checked,
        reextract: $("folder-reextract").checked,
      },
    });
    $("folder-modal").close();
    const recent = [path, ...JSON.parse(localStorage.getItem("recentFolders") || "[]").filter((p) => p !== path)].slice(0, 5);
    localStorage.setItem("recentFolders", JSON.stringify(recent));
    toast(`Found ${res.count} video(s) — extracting them one after another`, "ok");
    await refreshVideos();
    kickJobs();
  } catch (err) {
    toast(err.message, "error");
  }
}

// ---------------------------------------------------------------- import: single files

function extractFps() {
  const v = parseFloat($("folder-fps").value);
  return Number.isFinite(v) && v > 0 ? v : null;
}

async function onVideoPicked(e) {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  try {
    const probe = await api("/api/import/video/probe", { json: { filename: file.name } });
    if (!probe.extract_ready) return toast("Install ffmpeg or opencv-python to extract frames", "error");
    if (probe.exists && (probe.num_frames > 0 || probe.has_annotations)) {
      state.pendingImport = { type: "video", file, probe };
      showConflict(probe);
      return;
    }
    await uploadVideo(file, "create", probe.video_id);
  } catch (err) {
    toast(err.message, "error");
  }
}

async function onFramesPicked(e) {
  const files = [...(e.target.files || [])];
  e.target.value = "";
  if (!files.length) return;
  try {
    const stem = files.length === 1 ? files[0].name.replace(/\.zip$/i, "") : `frames_${files[0].name}`;
    const probe = await api("/api/import/video/probe", { json: { filename: stem } });
    if (probe.exists && (probe.num_frames > 0 || probe.has_annotations)) {
      state.pendingImport = { type: "frames", files, probe };
      showConflict(probe);
      return;
    }
    await uploadFrames(files, "create", probe.video_id);
  } catch (err) {
    toast(err.message, "error");
  }
}

async function onJsonPicked(e) {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file || !state.vid) return;
  await flushAll();
  const fd = new FormData();
  fd.append("file", file);
  fd.append("video_id", state.vid);
  fd.append("merge", "true");
  try {
    await api("/api/import/json", { body: fd });
    await openVideo(state.vid);
    toast("Annotations imported", "ok");
  } catch (err) {
    toast(err.message, "error");
  }
}

function showConflict(probe) {
  $("conflict-message").textContent = `"${probe.video_id}" is already in the library.`;
  $("conflict-stats").innerHTML = `<li>${probe.num_frames} frames</li>
    <li>${probe.labelled} labelled · ${probe.boxed} with a POI box</li>`;
  $("conflict-open").disabled = !(probe.num_frames > 0);
  $("conflict-modal").showModal();
}

async function onConflictSubmit(e) {
  e.preventDefault();
  const action = e.submitter?.value || "cancel";
  $("conflict-modal").close();
  const pending = state.pendingImport;
  if (!pending || action === "cancel") { state.pendingImport = null; return; }
  if (action === "open") {
    state.pendingImport = null;
    await refreshVideos();
    await openVideo(pending.probe.video_id);
    return;
  }
  if (action === "reextract_wipe") {
    $("wipe-id").textContent = pending.probe.video_id;
    $("wipe-confirm").value = "";
    $("wipe-go").disabled = true;
    $("wipe-modal").showModal();
    return;
  }
  await runPendingImport(action);
}

async function onWipeSubmit(e) {
  e.preventDefault();
  $("wipe-modal").close();
  if (e.submitter?.value !== "confirm") { state.pendingImport = null; return; }
  await runPendingImport("reextract_wipe");
}

async function runPendingImport(mode) {
  const p = state.pendingImport;
  state.pendingImport = null;
  if (!p) return;
  if (p.type === "video") await uploadVideo(p.file, mode, p.probe.video_id);
  else await uploadFrames(p.files, mode, p.probe.video_id);
}

async function uploadVideo(file, mode, videoId) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", mode);
  fd.append("video_id", videoId);
  const fps = extractFps();
  if (fps) fd.append("fps", String(fps));
  try {
    const res = await withWait(`Uploading ${file.name}…`, () => api("/api/import/video", { body: fd }));
    if (res.mode === "open") { await refreshVideos(); return openVideo(res.video_id); }
    await refreshVideos();
    kickJobs();
    const job = await waitForJob(res.job, `Extracting frames from ${file.name}`);
    if (job.status === "done") {
      await refreshVideos();
      await openVideo(res.video_id);
      toast(job.message, "ok");
    }
  } catch (err) {
    if (err.status === 409 && err.detail?.probe) {
      state.pendingImport = { type: "video", file, probe: err.detail.probe };
      showConflict(err.detail.probe);
      return;
    }
    toast(err.message, "error");
  }
}

async function uploadFrames(files, mode, videoId) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  fd.append("mode", mode);
  fd.append("video_id", videoId);
  try {
    const res = await withWait("Uploading frames…", () => api("/api/import/frames", { body: fd }));
    await refreshVideos();
    await openVideo(res.video_id);
    if (res.num_frames) toast(`Imported ${res.num_frames} frames`, "ok");
  } catch (err) {
    if (err.status === 409 && err.detail?.probe) {
      state.pendingImport = { type: "frames", files, probe: err.detail.probe };
      showConflict(err.detail.probe);
      return;
    }
    toast(err.message, "error");
  }
}

// ---------------------------------------------------------------- playback / keys

function togglePlay() {
  if (state.playing) return stopPlay();
  if (!state.frames.length) return;
  if (state.idx >= state.frames.length - 1) go(0);
  state.playing = true;
  $("btn-play").textContent = "⏸";
  const interval = 1000 / Math.min(30, state.video?.fps || 10);
  const step = async () => {
    if (!state.playing) return;
    const t0 = performance.now();
    if (state.idx >= state.frames.length - 1) return stopPlay();
    await go(state.idx + 1);
    setTimeout(step, Math.max(0, interval - (performance.now() - t0)));
  };
  step();
}

function stopPlay() {
  state.playing = false;
  $("btn-play").textContent = "▶";
}

function onKey(e) {
  if (e.target.matches("input, textarea, select") || dialogOpen()) return;
  if (!state.vid) return;
  const k = e.key;
  if (k === "ArrowLeft") { e.preventDefault(); go(state.idx - (e.shiftKey ? 10 : 1)); }
  else if (k === "ArrowRight") { e.preventDefault(); go(state.idx + (e.shiftKey ? 10 : 1)); }
  else if (k === "Home") { e.preventDefault(); go(0); }
  else if (k === "End") { e.preventDefault(); go(state.frames.length - 1); }
  else if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "PageUp") { e.preventDefault(); stepVideo(-1); }
  else if (k === "PageDown") { e.preventDefault(); stepVideo(1); }
  else if (k === "Delete" || k === "Backspace") { e.preventDefault(); if (state.selectedId) deleteObject(state.selectedId); }
  else if ((k === "z" || k === "Z") && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
  else if (e.ctrlKey || e.metaKey || e.altKey) return;
  else if ((k === "d" || k === "D") && e.shiftKey) { e.preventDefault(); runDetectTrack(); }
  else if (k === "d" || k === "D") { e.preventDefault(); runDetect(); }
  else if (k === "t" || k === "T") {
    e.preventDefault();
    if (e.shiftKey && state.trackOpts) runTrack(state.trackOpts);
    else openTrackModal();
  }
  else if (k === "f" || k === "F") { e.preventDefault(); fillUntilNext(); }
  else if (k === "p" || k === "P") { e.preventDefault(); if (state.selectedId) setPoi(state.selectedId); }
  else if (k === "Escape") { state.drag = null; state.selectedId = null; renderPeople(); draw(); }
  else if (k === "+" || k === "=") setZoom(state.zoom * 1.2);
  else if (k === "-" || k === "_") setZoom(state.zoom / 1.2);
  else if (k === "0") setZoom(1);
  else if (k === "?") $("help-modal").showModal();
  else if (k >= "1" && k <= "5") {
    const b = (state.config.behaviours || [])[+k - 1];
    if (b) toggleBehaviour(b);
  }
}

init().catch((e) => toast(e.message || String(e), "error"));
