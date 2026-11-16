/* In-browser recognition engine: a line-by-line port of the Python server path, so the whole app
   can run as a static page (GitHub Pages) with nothing sent anywhere.

   Python source of each part:
     preprocess, fill_gaps, smooth, hand_speed, motion_span, transform  -> ssl_recognition/features.py
     resample                                                          -> ssl_recognition/data.py
     Predictor.predict                                                 -> ssl_recognition/infer.py
     Segmenter, best_window, top3, Session.step                        -> app/server.py
   tests/test_engine_parity.py runs this file under Node and checks it against the Python code on
   the same inputs, so the two cannot drift apart.

   Works in the browser (window.SSLEngine) and in Node (module.exports). */
(function (root) {
  "use strict";

  // ---- feature layout (ssl_recognition/landmarks.py) ---------------------------------------
  const FEATURE_DIM = 155, COORD_DIM = 153, N_HAND = 21, N_FACE = 13;
  const LEFT = [27, 90], RIGHT = [90, 153], LEFT_FLAG = 153, RIGHT_FLAG = 154;
  const HANDS = [[LEFT, LEFT_FLAG], [RIGHT, RIGHT_FLAG]];
  const TRAJ_POINTS = [0, 4, 8, 12, 16, 20], TRAJ_DIM = 24;
  const FACE_REL_POINTS = [0, 4, 8], FACE_REFS = [0, 1], FACE_DIM = N_FACE * 3 + 24;
  const WRIST = 0, MIDDLE_MCP = 9;

  const copyRows = (x) => x.map((r) => Float64Array.from(r));
  const zeros = (n) => new Float64Array(n);

  // ---- features.py ---------------------------------------------------------------------------
  function fillGaps(x, maxGap = 8) {
    const y = copyRows(x);
    for (const [[a0, a1], flag] of HANDS) {
      const idx = [];
      for (let t = 0; t < y.length; t++) if (y[t][flag] > 0) idx.push(t);
      for (let k = 0; k + 1 < idx.length; k++) {
        const a = idx[k], b = idx[k + 1], gap = b - a - 1;
        if (gap > 0 && gap <= maxGap) {
          for (let j = 1; j <= gap; j++) {
            const w = j / (gap + 1);
            for (let d = a0; d < a1; d++) y[a + j][d] = y[a][d] * (1 - w) + y[b][d] * w;
            y[a + j][flag] = 1;
          }
        }
      }
    }
    return y;
  }

  function smooth(x, k = 3) {
    if (k <= 1 || x.length < 3) return x;
    const y = copyRows(x), half = Math.floor(k / 2), n = x.length;
    for (const [[a0, a1], flag] of [[[0, 27], null], ...HANDS]) {
      const present = x.map((r) => {
        if (flag !== null) return r[flag] > 0;
        let s = 0;
        for (let d = a0; d < a1; d++) s += Math.abs(r[d]);
        return s > 0;
      });
      for (let t = 0; t < n; t++) {
        if (!present[t]) continue;
        const lo = Math.max(0, t - half), hi = Math.min(n, t + half + 1);
        let cnt = 0;
        const acc = zeros(a1 - a0);
        for (let u = lo; u < hi; u++) {
          if (!present[u]) continue;
          cnt++;
          for (let d = a0; d < a1; d++) acc[d - a0] += x[u][d];
        }
        for (let d = a0; d < a1; d++) y[t][d] = acc[d - a0] / cnt;
      }
    }
    return y;
  }

  function handCentre(r, a0) {
    let cx = 0, cy = 0;
    for (let p = 0; p < N_HAND; p++) { cx += r[a0 + p * 3]; cy += r[a0 + p * 3 + 1]; }
    return [cx / N_HAND, cy / N_HAND];
  }

  function handSpeed(x) {
    const sp = zeros(x.length);
    for (const [[a0], flag] of HANDS) {
      for (let t = 1; t < x.length; t++) {
        if (!(x[t][flag] > 0 && x[t - 1][flag] > 0)) continue;
        const c1 = handCentre(x[t], a0), c0 = handCentre(x[t - 1], a0);
        sp[t] = Math.max(sp[t], Math.hypot(c1[0] - c0[0], c1[1] - c0[1]));
      }
    }
    return sp;
  }

  function motionSpan(x, thr = 0.01, minKeep = 0.6) {
    const n = x.length;
    if (n < 8 || thr <= 0) return [0, n];
    const sp = handSpeed(x), moving = [];
    for (let t = 0; t < n; t++) if (sp[t] > thr) moving.push(t);
    if (!moving.length) return [0, n];
    let a = Math.max(0, moving[0] - 1), b = Math.min(n, moving[moving.length - 1] + 2);
    const need = Math.ceil(n * minKeep);
    if (b - a < need) {
      const mid = Math.floor((a + b) / 2);
      a = Math.max(0, Math.min(mid - Math.floor(need / 2), n - need));
      b = a + need;
    }
    return [a, b];
  }

  function preprocess(x, cfg, face) {
    const pp = cfg.preprocess || {};
    if (cfg.fill_gaps) x = fillGaps(x, Number(cfg.fill_gaps));
    if (pp.smooth) x = smooth(x, Number(pp.smooth));
    if (pp.motion_trim) {
      const [a, b] = motionSpan(x, Number(pp.motion_trim));
      x = x.slice(a, b);
      if (face) face = face.slice(a, b);
    }
    return [x, face];
  }

  // ---- data.py ---------------------------------------------------------------------------------
  function resample(x, L) {
    const t = x.length;
    if (t === L) return copyRows(x);
    if (t === 1) return Array.from({ length: L }, () => Float64Array.from(x[0]));
    const step = (t - 1) / (L - 1), out = [];
    for (let i = 0; i < L; i++) {
      const src = i === L - 1 ? t - 1 : i * step;
      const lo = Math.floor(src), hi = Math.min(lo + 1, t - 1), w = src - lo;
      const r = zeros(x[0].length);
      for (let d = 0; d < r.length; d++) r[d] = x[lo][d] * (1 - w) + x[hi][d] * w;
      out.push(r);
    }
    return out;
  }

  // ---- features.py transform -------------------------------------------------------------------
  function transform(x, cfg, face) {
    const f = cfg.features || {};
    const T = x.length, out = [];
    const traj = f.trajectory ? trajectory(x) : null;
    for (let t = 0; t < T; t++) {
      const y = Float64Array.from(x[t]);
      if (f.hand_local) {
        for (const [[a0], flag] of HANDS) {
          if (!(y[flag] > 0)) { for (let d = a0; d < a0 + 63; d++) y[d] = 0; continue; }
          const wx = x[t][a0], wy = x[t][a0 + 1], wz = x[t][a0 + 2];
          let size = Math.hypot(x[t][a0 + MIDDLE_MCP * 3] - wx, x[t][a0 + MIDDLE_MCP * 3 + 1] - wy);
          if (size < 1e-4) size = 1;
          for (let p = 0; p < N_HAND; p++) {
            y[a0 + p * 3] = (x[t][a0 + p * 3] - wx) / size;
            y[a0 + p * 3 + 1] = (x[t][a0 + p * 3 + 1] - wy) / size;
            y[a0 + p * 3 + 2] = (x[t][a0 + p * 3 + 2] - wz) / size;
          }
        }
      }
      out.push(y);
    }
    const rows = [];
    for (let t = 0; t < T; t++) {
      const parts = [out[t]];
      if (f.velocity) {
        const v = zeros(COORD_DIM);
        if (t > 0) for (let d = 0; d < COORD_DIM; d++) v[d] = out[t][d] - out[t - 1][d];
        parts.push(v);
      }
      if (traj) parts.push(traj[t]);
      if (f.face) parts.push(faceBlockRow(x[t], face ? face[t] : null));
      const len = parts.reduce((s, p) => s + p.length, 0), r = new Float32Array(len);
      let k = 0;
      for (const p of parts) { r.set(p, k); k += p.length; }
      rows.push(r);
    }
    return rows;
  }

  function trajectory(x) {
    const T = x.length, out = Array.from({ length: T }, () => zeros(TRAJ_DIM));
    HANDS.forEach(([[a0], flag], k) => {
      let cx = 0, cy = 0, n = 0;
      for (let t = 0; t < T; t++) if (x[t][flag] > 0) { cx += x[t][a0]; cy += x[t][a0 + 1]; n++; }
      if (!n) return;
      cx /= n; cy /= n;
      for (let t = 0; t < T; t++) {
        if (!(x[t][flag] > 0)) continue;
        TRAJ_POINTS.forEach((p, j) => {
          out[t][k * 12 + j * 2] = x[t][a0 + p * 3] - cx;
          out[t][k * 12 + j * 2 + 1] = x[t][a0 + p * 3 + 1] - cy;
        });
      }
    });
    return out;
  }

  function faceBlockRow(xr, fr) {
    const out = zeros(FACE_DIM);
    if (!fr) return out;
    let s = 0;
    for (let d = 0; d < N_FACE * 3; d++) { out[d] = fr[d]; s += Math.abs(fr[d]); }
    const presentFace = s > 0;
    let k = N_FACE * 3;
    for (const [[a0], flag] of HANDS) {
      const ok = xr[flag] > 0 && presentFace;
      for (const ref of FACE_REFS) {
        for (const p of FACE_REL_POINTS) {
          out[k++] = ok ? xr[a0 + p * 3] - fr[ref * 3] : 0;
          out[k++] = ok ? xr[a0 + p * 3 + 1] - fr[ref * 3 + 1] : 0;
        }
      }
    }
    return out;
  }

  /** Predictor.predict without the network: (T, 155) + (T, 39) -> (seq_len, D) model input. */
  function modelInput(x, faceRows, cfg) {
    const L = Number(cfg.seq_len || 32);
    if ((cfg.features || {}).face) {
      const face = faceRows || x.map(() => zeros(N_FACE * 3));
      const [xp, fp] = preprocess(x, cfg, face);
      return transform(resample(xp, L), cfg, resample(fp, L));
    }
    const [xp] = preprocess(x, cfg, null);
    return transform(resample(xp, L), cfg, null);
  }

  // ---- server.py: segmentation and decoding ------------------------------------------------
  const REST_Y = 1.6, WRIST_Y = [16, 19], STILL_MS = 1000, STILL_THR = 0.035;
  const WINDOW_MARGIN = 0.15;

  function isActive(feat, restY = REST_Y) {
    let any = false;
    for (let d = 0; d < 27; d++) if (feat[d] !== 0) { any = true; break; }
    if (feat[LEFT_FLAG] + feat[RIGHT_FLAG] <= 0 || !any) return false;
    return Math.min(feat[WRIST_Y[0]], feat[WRIST_Y[1]]) < restY;
  }

  function handCentres(feat) {
    return [feat[LEFT_FLAG] > 0 ? handCentre(feat, LEFT[0]) : null,
            feat[RIGHT_FLAG] > 0 ? handCentre(feat, RIGHT[0]) : null];
  }

  class Segmenter {
    constructor({ gapMs = 700, minLen = 8, maxMs = 6000, restY = REST_Y, stillMs = STILL_MS,
                  stillThr = STILL_THR, minSignMs = 800 } = {}) {
      Object.assign(this, { gapMs, minLen, maxMs, restY, stillMs, stillThr, minSignMs });
      this.seg = []; this.faces = []; this.times = [];
      this.lastFace = null; this.startMs = null; this.lastActiveMs = null;
      this.last = null; this.stillProgress = 0; this.endReason = "";
    }

    stillRun() {
      const lo = [null, null], hi = [null, null];
      let start = this.seg.length;
      for (let i = this.seg.length - 1; i >= 0; i--) {
        if (!isActive(this.seg[i], this.restY)) break;
        const cs = handCentres(this.seg[i]);
        let moved = false;
        cs.forEach((c, h) => {
          if (!c) return;
          lo[h] = lo[h] ? [Math.min(lo[h][0], c[0]), Math.min(lo[h][1], c[1])] : c;
          hi[h] = hi[h] ? [Math.max(hi[h][0], c[0]), Math.max(hi[h][1], c[1])] : c;
          if (Math.hypot(hi[h][0] - lo[h][0], hi[h][1] - lo[h][1]) > this.stillThr) moved = true;
        });
        if (moved) break;
        start = i;
      }
      if (start >= this.seg.length) return [0, start];
      return [this.times[this.times.length - 1] - this.times[start], start];
    }

    push(feat, tMs, face) {
      this.stillProgress = 0;
      face = face || zeros(N_FACE * 3);
      if (isActive(feat, this.restY)) {
        if (!this.seg.length) this.startMs = tMs;
        this.seg.push(feat); this.faces.push(face); this.times.push(tMs);
        this.lastActiveMs = tMs;
        if (this.startMs !== null && tMs - this.startMs >= this.maxMs) return this.flush("time limit");
        if (this.stillMs && tMs - this.startMs >= this.minSignMs) {
          const [runMs, runStart] = this.stillRun();
          this.stillProgress = Math.min(1, runMs / this.stillMs);
          if (runMs >= this.stillMs) {
            const keepUntil = this.times[runStart] + 300;
            let cut = this.times.findIndex((t) => t > keepUntil);
            if (cut < 0) cut = this.seg.length;
            const keep = Math.max(cut, runStart + 1);
            this.seg = this.seg.slice(0, keep); this.times = this.times.slice(0, keep);
            this.faces = this.faces.slice(0, keep);
            return this.flush("hands still");
          }
        }
      } else if (this.seg.length) {
        this.seg.push(feat); this.faces.push(face); this.times.push(tMs);
        if (this.lastActiveMs !== null && tMs - this.lastActiveMs >= this.gapMs) return this.flush("hands lowered");
      }
      return null;
    }

    flush(reason = "done pressed") {
      const seg = this.seg, faces = this.faces;
      this.seg = []; this.times = []; this.faces = [];
      this.lastActiveMs = null; this.startMs = null; this.stillProgress = 0;
      while (seg.length && !isActive(seg[seg.length - 1], this.restY)) { seg.pop(); faces.pop(); }
      if (seg.length < this.minLen) return null;
      this.endReason = reason; this.last = seg; this.lastFace = faces;
      return seg;
    }
  }

  // linspace(a, b, num) then int(), as in best_window
  function linspaceInts(a, b, num) {
    const step = (b - a) / (num - 1), out = new Set();
    for (let i = 0; i < num; i++) out.add(Math.trunc(i === num - 1 ? b : a + i * step));
    return [...out].sort((p, q) => p - q);
  }

  /** Engine: loads the ONNX model with onnxruntime(-web|-node) and mirrors server.Session. */
  class Engine {
    constructor(ort, session, cfg, names, opts = {}) {
      this.ort = ort; this.session = session; this.cfg = cfg; this.names = names;
      this.inputName = session.inputNames[0];
      this.window = opts.window || 40; this.stride = opts.stride || 1;
      this.reset();
    }

    static async create(ort, modelSource, cfg, names, opts) {
      const session = await ort.InferenceSession.create(modelSource, opts && opts.sessionOptions);
      return new Engine(ort, session, cfg, names, opts);
    }

    reset() {
      this.buf = []; this.fbuf = []; this.seg = new Segmenter(); this.frameI = 0; this.top = [];
    }

    /** Softmax probabilities for a batch of (T, 155) clips. */
    async predictBatch(clips, faces) {
      const inputs = clips.map((x, i) => modelInput(x, faces ? faces[i] : null, this.cfg));
      const L = inputs[0].length, D = inputs[0][0].length, data = new Float32Array(inputs.length * L * D);
      inputs.forEach((rows, b) => rows.forEach((r, t) => data.set(r, (b * L + t) * D)));
      const tensor = new this.ort.Tensor("float32", data, [inputs.length, L, D]);
      const res = await this.session.run({ [this.inputName]: tensor });
      const logits = res[this.session.outputNames[0]].data, C = logits.length / inputs.length;
      return inputs.map((_, b) => softmax(logits.subarray(b * C, (b + 1) * C)));
    }

    async predict(x, face) { return (await this.predictBatch([x], [face]))[0]; }

    top3(p) {
      return [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3)
        .map((i) => ({ ...this.names[i], p: Math.round(p[i] * 1000) / 1000 }));
    }

    async bestWindow(x, face) {
      const n = x.length;
      if (n < 16) return [this.top3(await this.predict(x, face)), [0, n]];
      const minLen = Math.max(12, Math.floor(n * 0.4));
      const starts = linspaceInts(0, Math.floor(n * 0.45), 10);
      const ends = linspaceInts(Math.floor(n * 0.65), n, 7);
      const spans = [[0, n]];
      for (const s0 of starts) for (const e0 of ends) if (e0 - s0 >= minLen) spans.push([s0, e0]);
      const probs = await this.predictBatch(spans.map(([a, b]) => x.slice(a, b)),
                                            spans.map(([a, b]) => face.slice(a, b)));
      let best = probs[0], bestConf = Math.max(...probs[0]) + WINDOW_MARGIN, span = spans[0];
      for (let i = 1; i < spans.length; i++) {
        const m = Math.max(...probs[i]);
        if (m > bestConf) { best = probs[i]; bestConf = m; span = spans[i]; }
      }
      return [this.top3(best), span];
    }

    /** Same message in, same result out as the server's /ws_feat. */
    async step(feat, tMs, end = false, face = null) {
      feat = Float64Array.from(feat);
      face = face ? Float64Array.from(face) : zeros(N_FACE * 3);
      this.buf.push(feat); this.fbuf.push(face);
      if (this.buf.length > this.window) { this.buf.shift(); this.fbuf.shift(); }
      this.frameI++;
      if (this.frameI % this.stride === 0 && this.buf.length >= Math.floor(this.window / 2)) {
        const present = this.buf.filter((r) => r[LEFT_FLAG] + r[RIGHT_FLAG] > 0).length;
        this.top = present / this.buf.length > 0.5 ? this.top3(await this.predict(this.buf, this.fbuf)) : [];
      }
      const done = end ? this.seg.flush() : this.seg.push(feat, tMs, face);
      const out = { top: this.top, hands: feat[LEFT_FLAG] + feat[RIGHT_FLAG],
                    recording: this.seg.seg.length, still: Math.round(this.seg.stillProgress * 100) / 100 };
      if (done) {
        const [final, span] = await this.bestWindow(done, this.seg.lastFace);
        out.final = final; out.final_frames = span[1] - span[0];
        out.final_span = [span[0], span[1], done.length]; out.end_reason = this.seg.endReason;
      }
      return out;
    }
  }

  function softmax(z) {
    let m = -Infinity;
    for (const v of z) m = Math.max(m, v);
    const e = Array.from(z, (v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  const api = { Engine, Segmenter, fillGaps, smooth, handSpeed, motionSpan, preprocess, resample,
                transform, modelInput, isActive, linspaceInts, softmax, FEATURE_DIM, FACE_DIM };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SSLEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
