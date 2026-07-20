/* Trauma Behaviour Video Annotator — frontend */

const BEHAVIOUR_LABELS = {
  flashback: "Flashback",
  avoidance: "Avoidance",
  negative_emotion: "Negative emotion",
  hyper_arousal: "Hyperarousal",
  normal: "Normal",
};

const MIN_DETECT_CONF = 0.35;
const POI_COLOR = "#e11d2e";
/** 1-based id → color (POI always overrides to red). */
const ID_COLORS = [
  null,
  "#22c55e", // 1 green
  "#3b82f6", // 2 blue
  "#f59e0b", // 3 amber
  "#a855f7", // 4 purple
  "#06b6d4", // 5 cyan
  "#ec4899", // 6 pink
  "#84cc16", // 7 lime
  "#14b8a6", // 8 teal
  "#f97316", // 9 orange
  "#6366f1", // 10 indigo
];

function displayIdOf(obj) {
  const id = String(obj?.id || "");
  if (!id) return 0;
  if (!state.trackNumById[id]) {
    state.trackNumById[id] = state.nextTrackNum++;
  }
  return state.trackNumById[id];
}

function objectColor(obj) {
  if (obj.is_poi) return POI_COLOR;
  const n = obj.num_id || displayIdOf(obj) || 1;
  const idx = ((n - 1) % (ID_COLORS.length - 1)) + 1;
  return ID_COLORS[idx];
}

function assignNumIds() {
  // Stable: display number is tied to object.id, not row order
  state.objects.forEach((o) => {
    o.num_id = displayIdOf(o);
  });
}

function rebuildTrackNumsFromAnnotations() {
  state.trackNumById = {};
  state.nextTrackNum = 1;
  (state.frames || []).forEach((name) => {
    const objs = state.annotations?.[name]?.objects || [];
    objs.forEach((o) => {
      if (o?.id) displayIdOf(o);
    });
  });
}

const state = {
  config: null,
  videos: [],
  vid: null,
  frames: [],
  annotations: {},
  idx: 0,
  videoLabel: null,
  videoComment: "",
  context: "",
  hasUndo: false,
  image: null,
  /** Working object list for current frame (detections + confirmed). */
  objects: [],
  selectedId: null,
  draw: null,
  resize: null,
  zoom: 1,
  playTimer: null,
  saveTimer: null,
  pendingImport: null,
  trackOpts: null,
  /** Stable object.id → display number (same person keeps same ID across frames). */
  trackNumById: {},
  nextTrackNum: 1,
};

const $ = (id) => document.getElementById(id);

function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 3800);
}

function setStatus(msg) {
  $("save-status").textContent = msg;
}

let _waitDepth = 0;
function showWait(message = "Please wait…") {
  _waitDepth += 1;
  const overlay = $("wait-overlay");
  const msg = $("wait-message");
  if (msg) msg.textContent = message;
  if (overlay) overlay.classList.remove("hidden");
  document.body.style.cursor = "wait";
}

function hideWait() {
  _waitDepth = Math.max(0, _waitDepth - 1);
  if (_waitDepth > 0) return;
  const overlay = $("wait-overlay");
  if (overlay) overlay.classList.add("hidden");
  document.body.style.cursor = "";
}

async function withWait(message, fn) {
  showWait(message);
  try {
    return await fn();
  } finally {
    hideWait();
  }
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) body = await res.json();
  else body = await res.text();
  if (!res.ok) {
    const detail = body?.detail ?? body;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return body;
}

function uid() {
  return `obj_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---- Boot ----

async function init() {
  try {
    state.config = await api("/api/config");
  } catch (e) {
    toast(`Config failed: ${e.message}`, "error");
    state.config = { behaviours: Object.keys(BEHAVIOUR_LABELS), detect_defaults: { conf: 0.35 } };
  }

  const d = state.config.detect_defaults || {};
  const conf = Math.max(MIN_DETECT_CONF, d.conf != null ? Number(d.conf) : MIN_DETECT_CONF);
  if ($("detect-conf")) {
    $("detect-conf").value = conf;
    $("detect-conf-val").textContent = conf.toFixed(2);
  }
  if (d.imgsz && $("detect-imgsz")) $("detect-imgsz").value = String(d.imgsz);
  if (d.model && $("detect-model")) $("detect-model").value = d.model;
  if (d.augment != null && $("detect-augment")) $("detect-augment").checked = !!d.augment;

  bindUi();
  buildBehaviourButtons($("behaviour-btns"), "frame");
  buildBehaviourButtons($("obj-behaviour-btns"), "object");
  loadTrackOpts();
  await refreshVideos();
  if ($("detect-status")) {
    $("detect-status").textContent = state.config.yolo_ready ? "YOLO ready" : "YOLO missing";
  }
}

function buildBehaviourButtons(wrap, mode) {
  if (!wrap) return;
  wrap.innerHTML = "";
  (state.config?.behaviours || Object.keys(BEHAVIOUR_LABELS)).forEach((b, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip" + (b === "normal" ? " normal" : "");
    btn.dataset.behaviour = b;
    btn.textContent = mode === "frame" ? `${i + 1} ${BEHAVIOUR_LABELS[b] || b}` : (BEHAVIOUR_LABELS[b] || b);
    btn.addEventListener("click", () => {
      if (mode === "frame") toggleFrameBehaviour(b);
      else toggleObjBehaviour(b);
    });
    wrap.appendChild(btn);
  });
}

function bindUi() {
  $("btn-load-video").onclick = () => $("file-video").click();
  $("btn-load-frames").onclick = () => $("file-frames").click();
  $("btn-import-json").onclick = () => $("file-json").click();
  $("file-video").onchange = onVideoPicked;
  $("file-frames").onchange = onFramesPicked;
  $("file-json").onchange = onJsonPicked;

  $("btn-toggle-sidebar").onclick = () => {
    $("app").classList.add("sidebar-collapsed");
    $("btn-expand-sidebar").classList.remove("hidden");
    requestAnimationFrame(() => fitCanvas());
  };
  $("btn-expand-sidebar").onclick = () => {
    $("app").classList.remove("sidebar-collapsed");
    $("btn-expand-sidebar").classList.add("hidden");
    requestAnimationFrame(() => fitCanvas());
  };

  $("btn-first").onclick = () => go(0);
  $("btn-prev").onclick = () => go(state.idx - 1);
  $("btn-next").onclick = () => go(state.idx + 1);
  $("btn-last").onclick = () => go(state.frames.length - 1);
  $("btn-play").onclick = togglePlay;
  $("scrubber").oninput = (e) => go(+e.target.value);
  $("btn-zoom-in").onclick = () => setZoom(state.zoom * 1.2);
  $("btn-zoom-out").onclick = () => setZoom(state.zoom / 1.2);
  $("btn-zoom-fit").onclick = () => { state.zoom = 1; fitCanvas(); };

  $("detect-conf").oninput = (e) => {
    let v = Number(e.target.value);
    if (v < MIN_DETECT_CONF) {
      v = MIN_DETECT_CONF;
      e.target.value = v;
    }
    $("detect-conf-val").textContent = v.toFixed(2);
  };

  $("btn-detect").onclick = runDetect;
  $("btn-track").onclick = () => $("track-modal").showModal();
  $("btn-undo-track").onclick = undoTrack;
  $("btn-confirm-all").onclick = confirmAll;
  const btnConfirmTable = $("btn-confirm-all-table");
  if (btnConfirmTable) btnConfirmTable.onclick = confirmAll;
  $("btn-delete-unconfirmed").onclick = dropUnconfirmed;
  $("btn-clear-objects").onclick = clearObjects;
  $("btn-apply-objects").onclick = applyObjects;
  $("btn-set-poi").onclick = setSelectedAsPoi;
  $("btn-delete-obj").onclick = deleteSelected;
  $("obj-label").onchange = () => {
    const o = selectedObject();
    if (!o) return;
    o.label = $("obj-label").value;
    if (o.label === "person_of_interest") {
      setPoi(o.id);
    } else {
      o.is_poi = false;
      if (!state.objects.some((x) => x.is_poi) && state.objects.length) {
        const other = state.objects.find((x) => x.id !== o.id);
        if (other) {
          other.is_poi = true;
          other.label = "person_of_interest";
        }
      }
      renderObjectList({ skipFit: true });
      draw();
      schedulePersistObjects();
    }
  };

  $("btn-write-context").onclick = writeContext;
  $("btn-rebuild-coco").onclick = rebuildCoco;
  $("video-comment").onblur = saveVideoMeta;
  $("frame-comment").onblur = () => scheduleFrameSave({ comment: $("frame-comment").value });
  $("context").onblur = saveVideoMeta;

  document.querySelectorAll("#video-label-btns .chip").forEach((btn) => {
    btn.onclick = () => {
      const v = btn.dataset.vlabel;
      state.videoLabel = state.videoLabel === v ? null : v;
      syncVideoLabelUi();
      saveVideoMeta();
    };
  });

  $("conflict-form").onsubmit = onConflictSubmit;
  $("track-form").onsubmit = onTrackSubmit;
  $("wipe-form").onsubmit = onWipeSubmit;
  $("wipe-confirm").oninput = () => {
    $("wipe-go").disabled = $("wipe-confirm").value !== state.pendingImport?.probe?.video_id;
  };

  const canvas = $("canvas");
  canvas.addEventListener("mousedown", onCanvasDown);
  canvas.addEventListener("mousemove", onCanvasMove);
  canvas.addEventListener("mouseup", onCanvasUp);
  canvas.addEventListener("mouseleave", onCanvasUp);
  $("canvas-wrap").addEventListener("wheel", onWheelZoom, { passive: false });
  window.addEventListener("resize", () => fitCanvas());
  window.addEventListener("keydown", onKey);
}

// ---- Videos ----

async function refreshVideos(selectId) {
  const data = await api("/api/videos");
  state.videos = data.videos || [];
  const list = $("video-list");
  list.innerHTML = "";
  $("video-empty").classList.toggle("hidden", state.videos.length > 0);
  state.videos.forEach((v) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    if (v.id === state.vid) btn.classList.add("active");
    btn.innerHTML = `<strong>${escapeHtml(v.id)}</strong>
      <span class="meta">${v.num_frames}f · ${v.labelled} lab · ${v.boxed} box
      ${v.video_label ? "· " + v.video_label : ""}</span>`;
    btn.onclick = () => openVideo(v.id);
    li.appendChild(btn);
    list.appendChild(li);
  });
  if (selectId) await openVideo(selectId);
}

async function openVideo(vid) {
  stopPlay();
  const data = await api(`/api/video/${encodeURIComponent(vid)}`);
  state.vid = data.video_id;
  state.frames = data.frames || [];
  state.annotations = data.annotations || {};
  state.videoLabel = data.video_label;
  state.videoComment = data.video_comment || "";
  state.context = data.context || "";
  state.hasUndo = !!data.has_undo;
  state.idx = 0;
  state.zoom = 1;
  state.selectedId = null;
  rebuildTrackNumsFromAnnotations();

  $("empty-state").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  $("video-title").textContent = state.vid;
  $("video-comment").value = state.videoComment;
  $("context").value = state.context;
  $("scrubber").max = Math.max(0, state.frames.length - 1);
  $("btn-export-json").href = `/api/export/${encodeURIComponent(state.vid)}`;
  $("btn-export-coco").href = `/api/export-coco/${encodeURIComponent(state.vid)}`;
  $("btn-undo-track").disabled = !state.hasUndo;
  syncVideoLabelUi();
  await refreshVideos();
  await go(0);
}

function syncVideoLabelUi() {
  document.querySelectorAll("#video-label-btns .chip").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.vlabel === state.videoLabel);
  });
}

function currentSlot() {
  const name = state.frames[state.idx];
  if (!name) return { behaviours: [], comment: "", bbox: null, objects: [] };
  const slot = state.annotations[name] || { behaviours: [], comment: "", bbox: null, objects: [] };
  slot.behaviours = slot.behaviours || [];
  slot.objects = slot.objects || [];
  return slot;
}

function loadObjectsFromSlot() {
  const slot = currentSlot();
  const objs = (slot.objects || []).map((o) => ({
    id: o.id || uid(),
    bbox: o.bbox || o.box,
    box: o.bbox || o.box,
    label: o.label || (o.is_poi ? "person_of_interest" : "person"),
    behaviours: [...(o.behaviours || [])],
    confirmed: !!o.confirmed,
    source: o.source || "manual",
    conf: o.conf ?? null,
    is_poi: !!o.is_poi,
  }));
  if (!objs.length && slot.bbox) {
    objs.push({
      id: uid(),
      bbox: slot.bbox,
      box: slot.bbox,
      label: "person_of_interest",
      behaviours: [...(slot.behaviours || [])],
      confirmed: true,
      source: "manual",
      conf: null,
      is_poi: true,
    });
  }
  state.objects = objs;
  state.selectedId = objs.find((o) => o.is_poi)?.id || objs[0]?.id || null;
}

async function go(idx) {
  if (!state.frames.length) return;
  idx = Math.max(0, Math.min(state.frames.length - 1, idx));
  state.idx = idx;
  state.draw = null;
  $("scrubber").value = idx;
  $("frame-counter").textContent = `${idx + 1} / ${state.frames.length}`;
  $("frame-id").textContent = state.frames[idx];

  const slot = currentSlot();
  $("frame-comment").value = slot.comment || "";
  syncChipRow($("behaviour-btns"), slot.behaviours || []);
  loadObjectsFromSlot();
  renderObjectList();
  syncSelectionEditor();

  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = `/api/frame/${encodeURIComponent(state.vid)}/${idx}?t=${Date.now()}`;
  });
  state.image = img;
  fitCanvas();
  draw();
}

function syncChipRow(wrap, behaviours) {
  const set = new Set(behaviours || []);
  wrap.querySelectorAll(".chip").forEach((btn) => {
    btn.classList.toggle("active", set.has(btn.dataset.behaviour));
  });
}

function toggleFrameBehaviour(b) {
  const slot = currentSlot();
  let list = [...(slot.behaviours || [])];
  if (b === "normal") list = list.includes("normal") ? [] : ["normal"];
  else {
    list = list.filter((x) => x !== "normal");
    list = list.includes(b) ? list.filter((x) => x !== b) : [...list, b];
  }
  slot.behaviours = list;
  state.annotations[state.frames[state.idx]] = slot;
  syncChipRow($("behaviour-btns"), list);
  scheduleFrameSave({ behaviours: list });
}

function toggleObjBehaviour(b) {
  const o = selectedObject();
  if (!o) {
    toast("Select an object first (click a row or box)", "error");
    return;
  }
  let list = [...(o.behaviours || [])];
  if (b === "normal") list = list.includes("normal") ? [] : ["normal"];
  else {
    list = list.filter((x) => x !== "normal");
    list = list.includes(b) ? list.filter((x) => x !== b) : [...list, b];
  }
  o.behaviours = list;
  o.confirmed = true;
  // Mirror onto frame behaviours if this is the POI (optional convenience)
  syncChipRow($("obj-behaviour-btns"), list);
  updateObjectRowBehaviours(o.id);
  schedulePersistObjects();
}

function updateObjectRowBehaviours(objId) {
  const o = state.objects.find((x) => String(x.id) === String(objId));
  if (!o) return;
  const cell = document.querySelector(`#objects-tbody tr[data-oid="${String(objId).replace(/"/g, "")}"] .beh-cell`);
  if (!cell) {
    renderObjectList({ skipFit: true });
    syncSelectionEditor();
    return;
  }
  // Refresh checkbox states in-row without full table rebuild
  cell.querySelectorAll("input[data-beh]").forEach((inp) => {
    inp.checked = (o.behaviours || []).includes(inp.dataset.beh);
  });
  syncSelectionEditor();
}

function selectedObject() {
  return state.objects.find((o) => String(o.id) === String(state.selectedId)) || null;
}

// ---- Object list ----

function renderObjectList(opts = {}) {
  assignNumIds();
  const section = $("objects-table-section");
  const show = state.objects.length > 0;
  section.classList.toggle("hidden", !show);

  const confN = state.objects.filter((o) => o.confirmed).length;
  $("object-count").textContent = show
    ? `(${state.objects.length} total · ${confN} confirmed)`
    : "";

  const tbody = $("objects-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const behKeys = state.config?.behaviours || Object.keys(BEHAVIOUR_LABELS);

  state.objects.forEach((o) => {
    const idNum = o.num_id || displayIdOf(o);
    const color = objectColor(o);
    const tr = document.createElement("tr");
    tr.dataset.oid = String(o.id);
    if (String(o.id) === String(state.selectedId)) tr.classList.add("selected");
    if (o.is_poi) tr.classList.add("poi");
    if (!o.confirmed) tr.classList.add("pending");
    const confPct = o.conf != null ? `${Math.round(o.conf * 100)}%` : "—";

    const behChecks = behKeys.map((b) => {
      const short = (BEHAVIOUR_LABELS[b] || b)
        .replace("Negative emotion", "Neg.")
        .replace("Hyperarousal", "Hyper")
        .replace("Flashback", "Flash")
        .replace("Avoidance", "Avoid");
      const checked = (o.behaviours || []).includes(b) ? "checked" : "";
      return `<label class="beh-chip" title="${escapeHtml(BEHAVIOUR_LABELS[b] || b)}">
        <input type="checkbox" data-beh="${b}" ${checked} /> ${escapeHtml(short)}</label>`;
    }).join("");

    tr.innerHTML = `
      <td><span class="color-swatch" style="background:${color}"></span></td>
      <td><input type="checkbox" class="ok-check" ${o.confirmed ? "checked" : ""} title="Confirm" /></td>
      <td><strong>${idNum}</strong></td>
      <td>${o.is_poi ? '<span class="red">POI</span>' : "—"}</td>
      <td>
        <select class="row-label">
          <option value="person_of_interest" ${o.label === "person_of_interest" ? "selected" : ""}>Person of interest</option>
          <option value="person" ${o.label === "person" ? "selected" : ""}>Person</option>
          <option value="other" ${o.label === "other" ? "selected" : ""}>Other</option>
        </select>
      </td>
      <td>${confPct}</td>
      <td class="beh-cell">${behChecks}</td>
      <td>${escapeHtml(o.source || "")}${o.confirmed ? "" : " · pending"}</td>
      <td class="row-actions">
        <button type="button" class="btn-row-poi" title="Set as POI">POI</button>
        <button type="button" class="btn-row-edit">Edit</button>
        <button type="button" class="btn-row-del danger">Delete</button>
      </td>`;

    tr.onclick = (e) => {
      if (e.target.closest("button, select, input, label")) return;
      state.selectedId = o.id;
      document.querySelectorAll("#objects-tbody tr").forEach((r) => r.classList.toggle("selected", r === tr));
      syncSelectionEditor();
      draw();
    };

    tr.querySelector(".ok-check").onchange = (e) => {
      e.stopPropagation();
      o.confirmed = e.target.checked;
      if (o.confirmed && !state.objects.some((x) => x.is_poi)) {
        setPoi(o.id);
        return;
      }
      tr.classList.toggle("pending", !o.confirmed);
      schedulePersistObjects();
      draw();
    };

    tr.querySelectorAll(".beh-cell input[data-beh]").forEach((inp) => {
      inp.onchange = (e) => {
        e.stopPropagation();
        state.selectedId = o.id;
        const beh = inp.dataset.beh;
        let list = [...(o.behaviours || [])];
        if (beh === "normal") {
          list = inp.checked ? ["normal"] : [];
        } else {
          list = list.filter((x) => x !== "normal");
          if (inp.checked && !list.includes(beh)) list.push(beh);
          if (!inp.checked) list = list.filter((x) => x !== beh);
        }
        o.behaviours = list;
        o.confirmed = true;
        tr.querySelector(".ok-check").checked = true;
        tr.classList.remove("pending");
        syncChipRow($("obj-behaviour-btns"), list);
        if (beh === "normal" && inp.checked) {
          tr.querySelectorAll(".beh-cell input[data-beh]").forEach((x) => {
            if (x.dataset.beh !== "normal") x.checked = false;
          });
        } else if (beh !== "normal" && inp.checked) {
          const n = tr.querySelector('.beh-cell input[data-beh="normal"]');
          if (n) n.checked = false;
        }
        schedulePersistObjects();
        draw();
      };
    });

    tr.querySelector(".row-label").onchange = (e) => {
      e.stopPropagation();
      o.label = e.target.value;
      if (o.label === "person_of_interest") {
        setPoi(o.id);
      } else {
        o.is_poi = false;
        if (!state.objects.some((x) => x.is_poi) && state.objects.length) {
          const other = state.objects.find((x) => String(x.id) !== String(o.id));
          if (other) {
            other.is_poi = true;
            other.label = "person_of_interest";
            other.confirmed = true;
          }
        }
        ensureSinglePoi();
        renderObjectList({ skipFit: true });
        syncSelectionEditor();
        draw();
        schedulePersistObjects();
      }
    };

    tr.querySelector(".btn-row-poi").onclick = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      o.confirmed = true;
      setPoi(o.id);
    };
    tr.querySelector(".btn-row-edit").onclick = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      state.selectedId = o.id;
      document.querySelectorAll("#objects-tbody tr").forEach((r) => r.classList.toggle("selected", r === tr));
      syncSelectionEditor();
      draw();
      $("selected-object-panel")?.scrollIntoView({ block: "nearest" });
    };
    tr.querySelector(".btn-row-del").onclick = (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      deleteObject(o.id);
    };
    tbody.appendChild(tr);
  });

  // Never re-fit the video frame when rebuilding the table — that shrinks the page on click
  if (opts.fit) requestAnimationFrame(() => fitCanvas());
}

function syncSelectionEditor() {
  const o = selectedObject();
  $("no-selection").classList.toggle("hidden", !!o);
  $("selection-editor").classList.toggle("hidden", !o);
  if (!o) return;
  $("obj-label").value = o.label || "person";
  syncChipRow($("obj-behaviour-btns"), o.behaviours || []);
}

function ensureSinglePoi() {
  const pois = state.objects.filter((o) => o.is_poi);
  if (pois.length > 1) {
    const keep = pois[0];
    state.objects.forEach((o) => {
      o.is_poi = o.id === keep.id;
      if (!o.is_poi && o.label === "person_of_interest") o.label = "person";
    });
  }
  state.objects.forEach((o) => {
    if (o.is_poi) {
      o.label = "person_of_interest";
      o.confirmed = true;
    } else if (o.label === "person_of_interest") {
      o.label = "person";
      o.is_poi = false;
    }
  });
}

function boxArea(o) {
  const b = o.bbox || o.box;
  if (!b || b.length < 4) return 0;
  return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
}

function boxCenterScore(o) {
  const b = o.bbox || o.box;
  const imgW = state.image?.width || 1;
  const imgH = state.image?.height || 1;
  if (!b || b.length < 4) return 0;
  const cx = ((b[0] + b[2]) / 2) / imgW;
  const cy = ((b[1] + b[3]) / 2) / imgH;
  const d = Math.hypot(cx - 0.5, cy - 0.5);
  return Math.max(0, 1 - d * 1.4);
}

/** Score people for default POI: confidence + size + centrality. */
function scorePoiCandidate(o) {
  const imgW = state.image?.width || 1;
  const imgH = state.image?.height || 1;
  const conf = o.conf != null ? o.conf : 0.55;
  const areaN = Math.min(1, boxArea(o) / Math.max(1, imgW * imgH * 0.12));
  const center = boxCenterScore(o);
  return conf * 0.45 + areaN * 0.35 + center * 0.2;
}

function pickSmartPoiCandidate(objects) {
  const pool = (objects || []).filter((o) => (o.bbox || o.box));
  if (!pool.length) return null;
  let best = null;
  let bestScore = -1;
  pool.forEach((o) => {
    const s = scorePoiCandidate(o);
    if (s > bestScore) {
      bestScore = s;
      best = o;
    }
  });
  return best;
}

/**
 * Assign exactly one smart POI.
 * Keeps an existing POI unless force=true.
 * User can always change later via the POI button / label.
 */
function assignSmartPoi(opts = {}) {
  const existing = state.objects.find((o) => o.is_poi);
  if (existing && !opts.force) {
    ensureSinglePoi();
    return existing;
  }
  const among = opts.among || state.objects;
  const pick = pickSmartPoiCandidate(among);
  if (!pick) {
    ensureSinglePoi();
    return null;
  }
  state.objects.forEach((o) => {
    const is = String(o.id) === String(pick.id);
    o.is_poi = is;
    if (is) {
      o.label = "person_of_interest";
      o.confirmed = true;
    } else if (o.label === "person_of_interest") {
      o.label = "person";
    }
  });
  state.selectedId = pick.id;
  assignNumIds();
  return pick;
}

function dedupeObjectIds() {
  const seen = new Set();
  state.objects.forEach((o) => {
    let id = String(o.id || "");
    if (!id || seen.has(id)) {
      id = uid();
      o.id = id;
    }
    seen.add(id);
  });
}

/** Debounced save — avoids blocking UI on every click. */
let _persistTimer = null;
function schedulePersistObjects() {
  setStatus("Saving…");
  clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => persistObjects({ quiet: true }), 280);
}

/** Push current object list to the server so POI / delete / clear stick. */
async function persistObjects(opts = {}) {
  if (!state.vid || state.frames[state.idx] == null) return;
  dedupeObjectIds();
  ensureSinglePoi();
  assignNumIds();
  const objects = state.objects.map((o) => ({
    id: String(o.id),
    bbox: o.bbox || o.box,
    label: o.is_poi ? "person_of_interest" : (o.label === "person_of_interest" ? "person" : (o.label || "person")),
    behaviours: o.behaviours || [],
    confirmed: !!o.confirmed,
    source: o.source || "manual",
    conf: o.conf ?? null,
    is_poi: !!o.is_poi,
  }));
  const poi = objects.find((o) => o.is_poi);
  const body = {
    objects,
    bbox: poi?.bbox || null,
  };
  try {
    const res = await api(`/api/annotate/${encodeURIComponent(state.vid)}/${state.idx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    // Update cache only — do not rebuild the whole table (keeps UI responsive)
    state.annotations[res.frame] = res.annotation;
    setStatus("Saved");
    if (!opts.quiet) {
      syncSelectionEditor();
      draw();
    }
  } catch (e) {
    toast(`Save failed: ${e.message}`, "error");
    setStatus("Save failed");
  }
}

function setPoi(id) {
  id = String(id || "");
  if (!id) return;
  const target = state.objects.find((o) => String(o.id) === id);
  if (!target) {
    toast("Object not found", "error");
    return;
  }
  // Exactly one POI
  state.objects.forEach((o) => {
    const is = String(o.id) === id;
    o.is_poi = is;
    if (is) {
      o.label = "person_of_interest";
      o.confirmed = true;
    } else {
      if (o.label === "person_of_interest") o.label = "person";
      o.is_poi = false;
    }
  });
  state.selectedId = id;
  assignNumIds();
  renderObjectList({ skipFit: true });
  syncSelectionEditor();
  draw();
  schedulePersistObjects();
  toast(`POI → ID ${target.num_id || displayIdOf(target)} (red only)`, "ok");
}

function setSelectedAsPoi() {
  if (!state.selectedId) {
    toast("Select an object first", "error");
    return;
  }
  setPoi(state.selectedId);
}

function confirmAll() {
  state.objects.forEach((o) => { o.confirmed = true; });
  if (!state.objects.some((o) => o.is_poi)) {
    assignSmartPoi({ force: true });
  } else {
    ensureSinglePoi();
  }
  assignNumIds();
  renderObjectList({ skipFit: true });
  syncSelectionEditor();
  draw();
  schedulePersistObjects();
  const poi = state.objects.find((o) => o.is_poi);
  toast(
    poi
      ? `Confirmed — smart POI is ID ${poi.num_id || displayIdOf(poi)} (change anytime with POI)`
      : "Confirmed — object list ready under the frame",
    "ok",
  );
}

function dropUnconfirmed() {
  const before = state.objects.length;
  state.objects = state.objects.filter((o) => o.confirmed);
  if (!state.objects.find((o) => String(o.id) === String(state.selectedId))) {
    state.selectedId = state.objects[0] ? String(state.objects[0].id) : null;
  }
  ensureSinglePoi();
  assignNumIds();
  renderObjectList({ skipFit: true });
  syncSelectionEditor();
  draw();
  schedulePersistObjects();
  toast(`Dropped ${before - state.objects.length} unconfirmed`, "ok");
}

async function clearObjects() {
  if (!state.objects.length) return;
  if (!confirm(`Clear all ${state.objects.length} object(s) on this frame?`)) return;
  state.objects = [];
  state.selectedId = null;
  renderObjectList({ skipFit: true });
  syncSelectionEditor();
  draw();
  await persistObjects({ quiet: true });
  toast("All objects cleared on this frame", "ok");
}

function deleteObject(id) {
  id = String(id || "");
  if (!id) return;
  dedupeObjectIds();

  const victim = state.objects.find((o) => String(o.id) === id);
  if (!victim) {
    toast("Object not found", "error");
    return;
  }
  const wasPoi = !!victim.is_poi;
  const removedNum = victim.num_id || displayIdOf(victim);

  // Delete ONLY this object
  state.objects = state.objects.filter((o) => String(o.id) !== id);

  if (String(state.selectedId) === id) {
    state.selectedId = state.objects[0] ? String(state.objects[0].id) : null;
  }

  if (wasPoi && state.objects.length) {
    // Promote exactly one remaining person to POI
    const next = state.objects.find((o) => o.confirmed) || state.objects[0];
    state.objects.forEach((o) => {
      o.is_poi = String(o.id) === String(next.id);
      if (o.is_poi) {
        o.label = "person_of_interest";
        o.confirmed = true;
      } else if (o.label === "person_of_interest") {
        o.label = "person";
      }
    });
    state.selectedId = String(next.id);
  } else {
    ensureSinglePoi();
  }

  assignNumIds();
  renderObjectList({ skipFit: true });
  syncSelectionEditor();
  draw();
  schedulePersistObjects();
  toast(`Deleted object ID ${removedNum} — others kept`, "ok");
}

function deleteSelected() {
  if (!state.selectedId) {
    toast("Select an object to delete", "error");
    return;
  }
  deleteObject(state.selectedId);
}

async function applyObjects() {
  const keep = state.objects.filter((o) => o.confirmed);
  if (!keep.length) {
    toast("Confirm at least one object (OK checkbox) first", "error");
    return;
  }
  if (!keep.some((o) => o.is_poi)) {
    keep[0].is_poi = true;
    keep[0].label = "person_of_interest";
  }
  const n = Math.max(1, +$("apply-n").value || 1);
  const mode = ($("apply-mode")?.value === "merge") ? "merge" : "replace";
  try {
    await withWait(`Applying objects to ${n} frame(s) — please wait…`, async () => {
      setStatus("Applying…");
      const res = await api(`/api/apply-objects/${encodeURIComponent(state.vid)}/${state.idx}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objects: keep.map((o) => ({
            id: o.id,
            bbox: o.bbox || o.box,
            label: o.label,
            behaviours: o.behaviours || [],
            confirmed: true,
            source: o.source,
            conf: o.conf,
            is_poi: !!o.is_poi,
          })),
          num_frames: n,
          mode,
        }),
      });
      state.annotations[state.frames[state.idx]] = res.annotation;
      const data = await api(`/api/video/${encodeURIComponent(state.vid)}`);
      state.annotations = data.annotations;
      state.hasUndo = !!data.has_undo;
      $("btn-undo-track").disabled = !state.hasUndo;
      loadObjectsFromSlot();
      renderObjectList();
      syncSelectionEditor();
      draw();
      await refreshVideos();
      toast(`Applied ${keep.length} object(s) to ${res.covered} frame(s)`, "ok");
      setStatus("Applied");
    });
  } catch (e) {
    toast(e.message, "error");
    setStatus("Apply failed");
  }
}

// ---- Canvas ----

function fitCanvas() {
  const canvas = $("canvas");
  const wrap = $("canvas-wrap");
  const img = state.image;
  if (!img) return;
  const maxW = Math.max(320, wrap.clientWidth - 16);
  const maxH = Math.max(240, wrap.clientHeight - 16);
  const fit = Math.min(maxW / img.width, maxH / img.height);
  // Allow zoom above fit so the frame can be larger / clearer
  const scale = fit * state.zoom;
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  canvas._scale = scale;
  draw();
}

function setZoom(z) {
  state.zoom = Math.max(0.5, Math.min(4, z));
  fitCanvas();
}

function onWheelZoom(e) {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  setZoom(state.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
}

function toImage(x, y) {
  const s = $("canvas")._scale || 1;
  return { x: x / s, y: y / s };
}

function canvasPos(e) {
  const rect = $("canvas").getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function draw() {
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  const img = state.image;
  if (!img) return;
  const s = canvas._scale || 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  state.objects.forEach((o, i) => {
    const bb = o.bbox || o.box;
    if (!bb) return;
    const [x1, y1, x2, y2] = bb;
    const isSel = o.id === state.selectedId;
    const color = objectColor(o);
    ctx.lineWidth = isSel ? 3.5 : 2.5;
    ctx.strokeStyle = color;
    if (!o.confirmed) {
      ctx.setLineDash([7, 5]);
      ctx.globalAlpha = 0.85;
    } else {
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    if (o.is_poi) {
      ctx.fillStyle = "rgba(225,29,46,0.12)";
      ctx.fillRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.font = `bold ${Math.max(12, 13 * Math.min(s, 1.3))}px sans-serif`;
    const idNum = o.num_id || (i + 1);
    const tag = `ID ${idNum}${o.is_poi ? " POI" : ""}${o.conf != null ? " " + Math.round(o.conf * 100) + "%" : ""}${o.confirmed ? "" : " ?"}`;
    ctx.fillText(tag, x1 * s + 4, y1 * s + 16);
  });

  if (state.draw) {
    const b = state.draw;
    const x = Math.min(b.x1, b.x2) * s;
    const y = Math.min(b.y1, b.y2) * s;
    const w = Math.abs(b.x2 - b.x1) * s;
    const h = Math.abs(b.y2 - b.y1) * s;
    ctx.strokeStyle = "#e11d2e";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }

  // Handles for selected
  const sel = selectedObject();
  if (sel && (sel.bbox || sel.box) && !state.draw) {
    const [x1, y1, x2, y2] = sel.bbox || sel.box;
    const hs = 7;
    const col = objectColor(sel);
    [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([hx, hy]) => {
      ctx.fillStyle = col;
      ctx.fillRect(hx * s - hs / 2, hy * s - hs / 2, hs, hs);
    });
  }
}

function hitHandle(box, mx, my) {
  if (!box) return null;
  const s = $("canvas")._scale || 1;
  const pts = [
    ["nw", box[0], box[1]],
    ["ne", box[2], box[1]],
    ["sw", box[0], box[3]],
    ["se", box[2], box[3]],
  ];
  for (const [name, x, y] of pts) {
    if (Math.hypot(mx - x * s, my - y * s) < 12) return name;
  }
  return null;
}

function hitObject(imgPt) {
  // Top-most (last) first
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const o = state.objects[i];
    const bb = o.bbox || o.box;
    if (!bb) continue;
    if (imgPt.x >= bb[0] && imgPt.x <= bb[2] && imgPt.y >= bb[1] && imgPt.y <= bb[3]) return o;
  }
  return null;
}

function onCanvasDown(e) {
  if (!state.image) return;
  const p = canvasPos(e);
  const imgPt = toImage(p.x, p.y);

  const sel = selectedObject();
  if (sel && (sel.bbox || sel.box)) {
    const handle = hitHandle(sel.bbox || sel.box, p.x, p.y);
    if (handle) {
      const bb = sel.bbox || sel.box;
      state.resize = { handle, id: sel.id, box: { x1: bb[0], y1: bb[1], x2: bb[2], y2: bb[3] } };
      return;
    }
  }

  const hit = hitObject(imgPt);
  if (hit) {
    state.selectedId = hit.id;
    // Click unconfirmed detection → confirm it (opens bottom table)
    if (!hit.confirmed) {
      hit.confirmed = true;
      if (!state.objects.some((o) => o.is_poi)) {
        hit.is_poi = true;
        hit.label = "person_of_interest";
      }
      toast(`Confirmed ID ${hit.num_id || displayIdOf(hit)}`, "ok");
    }
    renderObjectList();
    syncSelectionEditor();
    draw();
    return;
  }

  state.draw = { x1: imgPt.x, y1: imgPt.y, x2: imgPt.x, y2: imgPt.y };
  draw();
}

function onCanvasMove(e) {
  const p = canvasPos(e);
  const imgPt = toImage(p.x, p.y);
  if (state.resize) {
    const b = { ...state.resize.box };
    const h = state.resize.handle;
    if (h.includes("n")) b.y1 = imgPt.y;
    if (h.includes("s")) b.y2 = imgPt.y;
    if (h.includes("w")) b.x1 = imgPt.x;
    if (h.includes("e")) b.x2 = imgPt.x;
    state.draw = b;
    draw();
    return;
  }
  if (state.draw) {
    state.draw.x2 = imgPt.x;
    state.draw.y2 = imgPt.y;
    draw();
  }
}

function onCanvasUp() {
  if (state.resize) {
    const b = state.draw || state.resize.box;
    const id = state.resize.id;
    state.resize = null;
    state.draw = null;
    const x1 = Math.min(b.x1, b.x2);
    const y1 = Math.min(b.y1, b.y2);
    const x2 = Math.max(b.x1, b.x2);
    const y2 = Math.max(b.y1, b.y2);
    const o = state.objects.find((x) => x.id === id);
    if (o && x2 - x1 > 4 && y2 - y1 > 4) {
      o.bbox = o.box = [x1, y1, x2, y2];
      o.confirmed = true;
      o.source = "manual";
    }
    renderObjectList();
    draw();
    return;
  }
  if (state.draw) {
    const b = state.draw;
    state.draw = null;
    const x1 = Math.min(b.x1, b.x2);
    const y1 = Math.min(b.y1, b.y2);
    const x2 = Math.max(b.x1, b.x2);
    const y2 = Math.max(b.y1, b.y2);
    if (x2 - x1 > 4 && y2 - y1 > 4) {
      const obj = {
        id: uid(),
        bbox: [x1, y1, x2, y2],
        box: [x1, y1, x2, y2],
        label: state.objects.some((o) => o.is_poi) ? "person" : "person_of_interest",
        behaviours: [],
        confirmed: true,
        source: "manual",
        conf: null,
        is_poi: !state.objects.some((o) => o.is_poi),
      };
      state.objects.push(obj);
      state.selectedId = obj.id;
      renderObjectList();
      syncSelectionEditor();
    }
    draw();
  }
}

// ---- Detect ----

async function runDetect() {
  try {
    await withWait("Detecting people — please wait…", async () => {
      setStatus("Detecting…");
      const body = {
        conf: Math.max(MIN_DETECT_CONF, +$("detect-conf").value),
        imgsz: +$("detect-imgsz").value,
        augment: $("detect-augment").checked,
        model: $("detect-model").value,
      };
      const res = await api(`/api/detect/${encodeURIComponent(state.vid)}/${state.idx}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const dets = (res.detections || []).filter((d) => (d.conf ?? 0) >= MIN_DETECT_CONF);
      const kept = state.objects.filter((o) => o.confirmed);
      const hadPoi = kept.some((o) => o.is_poi);
      const newObjs = dets.map((d) => ({
        id: uid(),
        bbox: d.box || d.bbox,
        box: d.box || d.bbox,
        label: "person",
        behaviours: [],
        confirmed: true, // ready to edit; user can uncheck OK if needed
        source: "detection",
        conf: d.conf,
        is_poi: false,
      }));
      state.objects = [...kept, ...newObjs];
      dedupeObjectIds();
      let poi = null;
      if (hadPoi) {
        ensureSinglePoi();
        poi = state.objects.find((o) => o.is_poi);
      } else if (state.objects.length) {
        poi = assignSmartPoi({ force: true, among: state.objects });
      }
      assignNumIds();
      state.selectedId = poi?.id || newObjs[0]?.id || state.objects[0]?.id || null;
      renderObjectList();
      syncSelectionEditor();
      requestAnimationFrame(() => fitCanvas());
      draw();
      schedulePersistObjects();
      if (!dets.length) {
        const tip = res.tip || "No detections ≥ 35%. Try a larger model or draw manually.";
        $("detect-tip").textContent = tip;
        toast(tip, "error");
      } else {
        const poiId = poi ? (poi.num_id || displayIdOf(poi)) : "?";
        $("detect-tip").textContent =
          `${dets.length} person(s). Smart POI → ID ${poiId} (red). Click POI on another row anytime to change.`;
        toast(
          hadPoi
            ? `${dets.length} found — kept your existing POI`
            : `${dets.length} found — smart POI set to ID ${poiId} (change anytime)`,
          "ok",
        );
      }
      setStatus(dets.length ? `${dets.length} detections` : "None ≥35%");
    });
  } catch (e) {
    toast(e.message, "error");
    setStatus("Detect failed");
  }
}

// ---- Save ----

function scheduleFrameSave(patch) {
  clearTimeout(state.saveTimer);
  setStatus("Saving…");
  state.saveTimer = setTimeout(() => saveFrame(patch), 280);
}

async function saveFrame(patch) {
  if (!state.vid) return;
  const slot = currentSlot();
  const body = {
    behaviours: patch.behaviours ?? slot.behaviours,
    comment: patch.comment ?? slot.comment,
  };
  if (patch.objects) body.objects = patch.objects;
  try {
    const res = await api(`/api/annotate/${encodeURIComponent(state.vid)}/${state.idx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.annotations[res.frame] = res.annotation;
    setStatus("Saved");
  } catch (e) {
    setStatus("Save failed");
    toast(e.message, "error");
  }
}

async function saveVideoMeta() {
  if (!state.vid) return;
  state.videoComment = $("video-comment").value;
  state.context = $("context").value;
  try {
    await api(`/api/video-meta/${encodeURIComponent(state.vid)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_label: state.videoLabel || "",
        video_comment: state.videoComment,
        context: state.context,
      }),
    });
    setStatus("Saved");
    await refreshVideos();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function writeContext() {
  await saveVideoMeta();
  const res = await api(`/api/write-context/${encodeURIComponent(state.vid)}`, { method: "POST" });
  toast(`Wrote ${res.file}`, "ok");
}

async function rebuildCoco() {
  const res = await api("/api/write-coco-all", { method: "POST" });
  toast(`Rebuilt COCO for ${res.count} videos`, "ok");
}

// ---- Track ----

function loadTrackOpts() {
  try {
    const raw = localStorage.getItem("trackOpts");
    if (raw) state.trackOpts = JSON.parse(raw);
  } catch { /* ignore */ }
}

function saveTrackOpts(opts) {
  state.trackOpts = opts;
  localStorage.setItem("trackOpts", JSON.stringify(opts));
}

function readTrackForm() {
  const f = $("track-form");
  const range = f.range.value;
  return {
    propagate_bbox: f.propagate_bbox.checked,
    propagate_behaviours: f.propagate_behaviours.checked,
    propagate_comment: f.propagate_comment.checked,
    overwrite_bbox: f.overwrite_bbox.checked,
    overwrite_labels: f.overwrite_labels.checked,
    stop_on_label_change: f.stop_on_label_change.checked,
    num_frames: range === "end" ? 0 : Math.max(1, +f.num_frames.value || 30),
    max_gap: +f.max_gap.value || 8,
    iou_thresh: +f.iou_thresh.value || 0.15,
  };
}

async function onTrackSubmit(e) {
  e.preventDefault();
  const action = e.submitter?.value || "cancel";
  $("track-modal").close();
  if (action === "cancel") return;
  const opts = readTrackForm();
  saveTrackOpts(opts);
  await runTrack(opts, action === "preview");
}

async function runTrack(opts, dryRun) {
  const poi = state.objects.find((o) => o.is_poi) || state.objects.find((o) => o.confirmed);
  const seed = poi?.bbox || poi?.box || currentSlot().bbox;
  const body = { ...opts, seed_box: seed || null, dry_run: !!dryRun };
  if (opts.propagate_bbox && !seed) {
    toast("Set a POI box first (Detect → confirm → Set as POI, or draw)", "error");
    return;
  }
  try {
    await withWait(dryRun ? "Previewing track — please wait…" : "Tracking forward — please wait…", async () => {
      setStatus(dryRun ? "Previewing…" : "Tracking…");
      const res = await api(`/api/track/${encodeURIComponent(state.vid)}/${state.idx}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (dryRun) {
        toast(`Preview: ${res.covered} frames (stop: ${res.stop_reason})`, "ok");
        return;
      }
      const data = await api(`/api/video/${encodeURIComponent(state.vid)}`);
      state.annotations = data.annotations;
      state.hasUndo = !!data.has_undo;
      $("btn-undo-track").disabled = !state.hasUndo;
      rebuildTrackNumsFromAnnotations();
      loadObjectsFromSlot();
      renderObjectList({ skipFit: true });
      draw();
      await refreshVideos();
      toast(
        `Tracked ${res.covered} frames · stable person ID kept · stopped: ${res.stop_reason}`,
        "ok",
      );
      if (res.stop_reason === "lost" && res.stopped_at != null) await go(res.stopped_at);
    });
  } catch (e) {
    toast(e.message, "error");
  }
}

async function undoTrack() {
  try {
    const res = await api(`/api/undo-track/${encodeURIComponent(state.vid)}`, { method: "POST" });
    const data = await api(`/api/video/${encodeURIComponent(state.vid)}`);
    state.annotations = data.annotations;
    state.hasUndo = false;
    $("btn-undo-track").disabled = true;
    loadObjectsFromSlot();
    renderObjectList();
    syncChipRow($("behaviour-btns"), currentSlot().behaviours || []);
    $("frame-comment").value = currentSlot().comment || "";
    draw();
    toast(`Restored ${res.restored} frames`, "ok");
  } catch (e) {
    toast(e.message, "error");
  }
}

// ---- Import (unchanged flow) ----

function extractFps() {
  const v = parseFloat($("extract-fps").value);
  return Number.isFinite(v) && v > 0 ? v : null;
}

async function onVideoPicked(e) {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  try {
    const probe = await api("/api/import/video/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name }),
    });
    if (!probe.extract_ready) {
      toast("Install opencv-python or ffmpeg to extract frames", "error");
      return;
    }
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
    const probe = await api("/api/import/video/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: files[0].name.replace(/\.zip$/i, "") }),
    });
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
  if (!file) return;
  if (!state.vid) {
    toast("Open a video first, then import JSON", "error");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("video_id", state.vid);
  fd.append("merge", "true");
  try {
    const res = await api("/api/import/json", { method: "POST", body: fd });
    await openVideo(res.video_id);
    toast("JSON imported", "ok");
  } catch (err) {
    toast(err.message, "error");
  }
}

function showConflict(probe) {
  $("conflict-message").textContent =
    `"${probe.video_id}" already exists. Choose how to continue.`;
  $("conflict-stats").innerHTML = `
    <li>${probe.num_frames} frames</li>
    <li>${probe.has_annotations ? "Has annotations" : "No annotations"}</li>
    <li>${probe.labelled} labelled · ${probe.boxed} boxed</li>`;
  $("conflict-open").disabled = !(probe.num_frames > 0);
  $("conflict-modal").showModal();
}

async function onConflictSubmit(e) {
  e.preventDefault();
  const action = e.submitter?.value || "cancel";
  $("conflict-modal").close();
  const pending = state.pendingImport;
  if (!pending || action === "cancel") {
    state.pendingImport = null;
    return;
  }
  if (action === "open") {
    await openVideo(pending.probe.video_id);
    state.pendingImport = null;
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
  const action = e.submitter?.value || "cancel";
  $("wipe-modal").close();
  if (action !== "confirm") {
    state.pendingImport = null;
    return;
  }
  await runPendingImport("reextract_wipe");
}

async function runPendingImport(mode) {
  const pending = state.pendingImport;
  state.pendingImport = null;
  if (!pending) return;
  if (pending.type === "video") await uploadVideo(pending.file, mode, pending.probe.video_id);
  else await uploadFrames(pending.files, mode, pending.probe.video_id);
}

async function uploadVideo(file, mode, videoId) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", mode);
  fd.append("video_id", videoId);
  const fps = extractFps();
  if (fps) fd.append("fps", String(fps));
  try {
    await withWait("Uploading & extracting frames — please wait…", async () => {
      const res = await api("/api/import/video", { method: "POST", body: fd });
      toast(`Extracted ${res.num_frames} frames`, "ok");
      await refreshVideos(res.video_id);
    });
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
    await withWait("Uploading frames — please wait…", async () => {
      const res = await api("/api/import/frames", { method: "POST", body: fd });
      toast(`Imported ${res.num_frames} frames`, "ok");
      await refreshVideos(res.video_id);
    });
  } catch (err) {
    if (err.status === 409 && err.detail?.probe) {
      state.pendingImport = { type: "frames", files, probe: err.detail.probe };
      showConflict(err.detail.probe);
      return;
    }
    toast(err.message, "error");
  }
}

// ---- Playback / keys ----

function togglePlay() {
  if (state.playTimer) { stopPlay(); return; }
  $("btn-play").textContent = "⏸";
  state.playTimer = setInterval(() => {
    if (state.idx >= state.frames.length - 1) { stopPlay(); return; }
    go(state.idx + 1);
  }, 120);
}

function stopPlay() {
  if (state.playTimer) clearInterval(state.playTimer);
  state.playTimer = null;
  $("btn-play").textContent = "▶";
}

function onKey(e) {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); go(state.idx - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); go(state.idx + 1); }
  else if (e.key === "Home") { e.preventDefault(); go(0); }
  else if (e.key === "End") { e.preventDefault(); go(state.frames.length - 1); }
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Delete" || e.key === "Backspace") {
    // Don't hijack when typing / using table controls; only delete selected object
    if (e.target.closest("input, textarea, select, button")) return;
    e.preventDefault();
    deleteSelected();
  }
  else if (e.key === "d" || e.key === "D") { e.preventDefault(); runDetect(); }
  else if (e.key === "a" || e.key === "A") { e.preventDefault(); applyObjects(); }
  else if (e.key === "t" || e.key === "T") {
    e.preventDefault();
    if (e.shiftKey && state.trackOpts) runTrack(state.trackOpts, false);
    else $("track-modal").showModal();
  }
  else if (e.key === "+" || e.key === "=") setZoom(state.zoom * 1.15);
  else if (e.key === "-" || e.key === "_") setZoom(state.zoom / 1.15);
  else if (e.key === "0") { state.zoom = 1; fitCanvas(); }
  else if (e.key >= "1" && e.key <= "5") {
    const list = state.config?.behaviours || [];
    const b = list[+e.key - 1];
    if (b) toggleFrameBehaviour(b);
  }
}

init().catch((e) => toast(e.message || String(e), "error"));
