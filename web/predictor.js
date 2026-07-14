/* ============================================================
   Golf Scoring Model — client-side predictor
   ------------------------------------------------------------
   The model is a plain Linear Regression trained in Python
   (see ../03_train_baseline.py and ../app.py). A linear model is
   just a weighted sum, so we can carry the learned numbers over
   to the browser and reproduce its predictions exactly — no
   server, no ML library needed here.

       predicted_score = intercept + Σ (weightᵢ × featureᵢ)

   The coefficients, intercept, and the Random Forest feature
   importances below were exported from the models fit on all
   1,678 clean player-seasons.
   ============================================================ */

const MODEL = {
  intercept: 69.56612208,
  coef: {
    avgDistance:  -0.02438961,
    fairwayPct:   -0.02202869,
    gir:          -0.16672618,
    avgPutts:      0.79466920,
    scrambling:   -0.04173923,
  },
  // Random Forest importances — how much of the prediction each stat drives.
  importance: {
    gir:          0.34776,
    scrambling:   0.25088,
    avgPutts:     0.22305,
    avgDistance:  0.11064,
    fairwayPct:   0.06767,
  },
  scoreRange: { best: 68.7, worst: 74.4 },
};

/* Feature definitions: order, labels, slider bounds, and a plain-language
   hint. Bounds are padded slightly beyond the real data range so the tour
   average sits comfortably in the middle of each slider. */
const FEATURES = [
  { key: "avgDistance", label: "Driving distance", hint: "average yards off the tee",
    min: 262, max: 322, step: 0.5, unit: "yds", decimals: 1 },
  { key: "fairwayPct",  label: "Driving accuracy", hint: "fairways hit",
    min: 42, max: 78, step: 0.5, unit: "%", decimals: 1 },
  { key: "gir",         label: "Greens in regulation", hint: "greens hit in regulation",
    min: 52, max: 74, step: 0.5, unit: "%", decimals: 1 },
  { key: "avgPutts",    label: "Putting", hint: "average putts per round",
    min: 27.5, max: 31, step: 0.05, unit: "", decimals: 2 },
  { key: "scrambling",  label: "Scrambling", hint: "par saves after missing the green",
    min: 43, max: 70, step: 0.5, unit: "%", decimals: 1 },
];

/* Tour-average starting point (dataset means). */
const TOUR_AVERAGE = {
  avgDistance: 290.8, fairwayPct: 61.4, gir: 65.7, avgPutts: 29.2, scrambling: 58.1,
};

/* Preset archetypes — realistic stat profiles built from the data's spread,
   not real individuals, so the model's behaviour is easy to explore. */
const PRESETS = [
  { id: "avg",     name: "Tour average", desc: "the middle of the pack",
    stats: { ...TOUR_AVERAGE } },
  { id: "bomber",  name: "The Bomber", desc: "long but wild",
    stats: { avgDistance: 312, fairwayPct: 54, gir: 66, avgPutts: 29.2, scrambling: 57 } },
  { id: "surgeon", name: "The Surgeon", desc: "accurate approach play",
    stats: { avgDistance: 286, fairwayPct: 71, gir: 71, avgPutts: 28.7, scrambling: 63 } },
  { id: "elite",   name: "Elite all-round", desc: "best-in-class everywhere",
    stats: { avgDistance: 300, fairwayPct: 66, gir: 71.5, avgPutts: 28.4, scrambling: 64 } },
];

/* ---- State ------------------------------------------------ */
const state = { ...TOUR_AVERAGE };
let displayedScore = MODEL.intercept; // last rendered number (for animation)

/* ---- Core prediction -------------------------------------- */
function predict(s) {
  const c = MODEL.coef;
  return (
    MODEL.intercept +
    c.avgDistance * s.avgDistance +
    c.fairwayPct  * s.fairwayPct +
    c.gir         * s.gir +
    c.avgPutts    * s.avgPutts +
    c.scrambling  * s.scrambling
  );
}

/* ---- Small helpers ---------------------------------------- */
const $  = (sel, root = document) => root.querySelector(sel);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function fmt(value, decimals) {
  return value.toFixed(decimals);
}

/* ---- Build sliders ---------------------------------------- */
function buildSliders() {
  const host = $("#sliders");
  FEATURES.forEach((f) => {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-top">
        <label class="slider-label" for="range-${f.key}">
          ${f.label}
          <span class="slider-hint">${f.hint}</span>
        </label>
        <span class="slider-value">
          <span id="val-${f.key}">${fmt(state[f.key], f.decimals)}</span><span class="slider-unit">${f.unit}</span>
        </span>
      </div>
      <input type="range" id="range-${f.key}"
             min="${f.min}" max="${f.max}" step="${f.step}"
             value="${state[f.key]}"
             aria-label="${f.label}, ${f.hint}" />
    `;
    host.appendChild(row);

    const input = $(`#range-${f.key}`, row);
    input.addEventListener("input", () => {
      state[f.key] = parseFloat(input.value);
      $(`#val-${f.key}`, row).textContent = fmt(state[f.key], f.decimals);
      paintTrack(input, f);
      clearActivePreset();
      update();
    });
    paintTrack(input, f);
  });
}

/* Colour the filled portion of a slider track up to the thumb. */
function paintTrack(input, f) {
  const pct = clamp01((state[f.key] - f.min) / (f.max - f.min)) * 100;
  input.style.setProperty("--fill", pct.toFixed(1) + "%");
}

/* ---- Build presets ---------------------------------------- */
function buildPresets() {
  const host = $(".presets");
  PRESETS.forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset";
    btn.dataset.id = p.id;
    btn.setAttribute("aria-pressed", "false");
    btn.innerHTML = `
      <span class="preset-name">${p.name}</span>
      <span class="preset-desc">${p.desc}</span>
    `;
    btn.addEventListener("click", () => applyPreset(p));
    host.appendChild(btn);
  });
}

function applyPreset(p) {
  Object.assign(state, p.stats);
  syncControls();
  document.querySelectorAll(".preset").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.id === p.id))
  );
  update();
}

function clearActivePreset() {
  document.querySelectorAll(".preset").forEach((b) =>
    b.setAttribute("aria-pressed", "false")
  );
}

/* Push state back into every slider (after a preset / reset). */
function syncControls() {
  FEATURES.forEach((f) => {
    const input = $(`#range-${f.key}`);
    input.value = state[f.key];
    $(`#val-${f.key}`).textContent = fmt(state[f.key], f.decimals);
    paintTrack(input, f);
  });
}

/* ---- Build feature-importance chart ----------------------- */
function buildChart() {
  const host = $("#importance");
  const entries = Object.entries(MODEL.importance).sort((a, b) => b[1] - a[1]);
  const max = entries[0][1];
  const labels = {
    gir: "Greens in regulation", scrambling: "Scrambling", avgPutts: "Putting",
    avgDistance: "Driving distance", fairwayPct: "Driving accuracy",
  };

  entries.forEach(([key, val], i) => {
    const row = document.createElement("div");
    row.className = "imp-row" + (i === 0 ? " lead" : "");
    row.innerHTML = `
      <div class="imp-head">
        <span class="imp-name">${labels[key]}${i === 0 ? '<span class="imp-note">most influential</span>' : ""}</span>
        <span class="imp-value">${Math.round(val * 100)}%</span>
      </div>
      <div class="imp-track"><span class="imp-bar" data-w="${(val / max) * 100}"></span></div>
    `;
    host.appendChild(row);
  });

  // Animate bars in on first paint (unless reduced motion).
  requestAnimationFrame(() => {
    host.querySelectorAll(".imp-bar").forEach((bar) => {
      bar.style.width = bar.dataset.w + "%";
    });
  });
}

/* ---- Update outputs (number + gauge) ---------------------- */
function update() {
  const score = predict(state);
  animateScore(displayedScore, score);
  displayedScore = score;

  // Gauge marker: position across the dataset's best→worst range.
  const { best, worst } = MODEL.scoreRange;
  const pos = clamp01((score - best) / (worst - best)) * 100;
  $("#gauge-marker").style.left = pos.toFixed(1) + "%";

  // A short contextual read-out.
  $("#hero-context").innerHTML = contextLine(score, pos);
}

function contextLine(score, pos) {
  let band;
  if (pos < 20) band = "elite — near the best seasons in the data";
  else if (pos < 45) band = "well above tour average";
  else if (pos < 60) band = "around tour average";
  else if (pos < 80) band = "below tour average";
  else band = "a season to forget";
  return `That&rsquo;s <strong>${band}</strong>.`;
}

/* Smoothly count the hero number between two values. */
function animateScore(from, to) {
  const el = $("#score");
  if (prefersReducedMotion || Math.abs(to - from) < 0.005) {
    el.textContent = to.toFixed(2);
    return;
  }
  const duration = 380;
  const start = performance.now();
  function frame(now) {
    const t = clamp01((now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    el.textContent = (from + (to - from) * eased).toFixed(2);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ---- Theme toggle ----------------------------------------- */
function initTheme() {
  const toggle = $("#theme-toggle");
  const label = $(".theme-toggle-label", toggle);
  const stored = localStorage.getItem("gsm-theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  const currentIsDark = () => {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr) return attr === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const refresh = () => {
    label.textContent = currentIsDark()
      ? label.dataset.labelDark : label.dataset.labelLight;
  };
  toggle.addEventListener("click", () => {
    const next = currentIsDark() ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("gsm-theme", next);
    refresh();
  });
  refresh();
}

/* ---- Boot ------------------------------------------------- */
buildPresets();
buildSliders();
buildChart();
initTheme();
applyPreset(PRESETS[0]); // start on "Tour average", marked selected

$("#reset").addEventListener("click", () => applyPreset(PRESETS[0]));
