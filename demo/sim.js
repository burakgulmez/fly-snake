// Snake engine + sensory features + rate network, a line-by-line port of
// src/snake/engine.py, src/baselines/heuristics.py and src/brain/policies.py.
// Works in the browser (global `FlySnake`) and in Node (module.exports).
(function (root) {
  "use strict";
  const WALL = 1, BODY = 2, TRAPPED = 3, TIMEOUT = 4, WIN = 5;
  const CAUSES = ["alive", "wall", "body", "trapped", "timeout", "win"];
  const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
  const INIT_LEN = 3;

  // ---- xorshift32 (bit-exact with the numba version) ----
  function rngNext(x) {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x >>> 0;
  }
  function rngSeed(seed) {
    // (seed * 2654435761 + 12345) mod 2^32, exact for seed < 2^53 via BigInt
    let x = Number((BigInt(seed) * 2654435761n + 12345n) & 0xFFFFFFFFn);
    if (x === 0) x = 1;
    for (let i = 0; i < 4; i++) x = rngNext(x);
    return x;
  }

  class Game {
    constructor(G, seed, timeoutK) {
      const A = G * G;
      this.G = G; this.A = A;
      this.body = new Int32Array(A);
      this.occ = new Uint8Array(A);
      this.dir = 0; this.rng = rngSeed(seed);
      this.tlimit = timeoutK * A;
      this.steps = 0; this.since = 0; this.score = 0; this.done = false; this.cause = 0;
      const hr = Math.floor(G / 2), hc = Math.floor(G / 2);
      for (let i = 0; i < INIT_LEN; i++) {
        const cell = (hr + (INIT_LEN - 1 - i)) * G + hc;
        this.body[i] = cell; this.occ[cell] = 1;
      }
      this.head = INIT_LEN - 1; this.len = INIT_LEN; this.food = -1;
      this.placeFood();
    }
    placeFood() {
      const free = this.A - this.len;
      if (free <= 0) { this.food = -1; return; }
      this.rng = rngNext(this.rng);
      let r = this.rng % free;
      for (let c = 0; c < this.A; c++) {
        if (this.occ[c] === 0) { if (r === 0) { this.food = c; return; } r--; }
      }
    }
    seg(i) { return this.body[(this.head - i + this.A) % this.A]; }
    absDir(a) { return (this.dir + a - 1 + 4) % 4; }
    target(d) {
      const h = this.seg(0), G = this.G;
      const r = Math.floor(h / G) + DR[d], c = (h % G) + DC[d];
      if (r < 0 || r >= G || c < 0 || c >= G) return -1;
      return r * G + c;
    }
    moveOutcome(a) {
      const cell = this.target(this.absDir(a));
      if (cell < 0) return WALL;
      if (this.occ[cell] === 1) {
        const tail = this.seg(this.len - 1);
        const eat = cell === this.food;
        if (!(cell === tail && !eat)) return BODY;
      }
      return 0;
    }
    step(a) {
      if (this.done) return 0;
      const out = this.moveOutcome(a);
      this.steps++;
      if (out !== 0) {
        let forced = true;
        for (let b = 0; b < 3; b++) if (this.moveOutcome(b) === 0) { forced = false; break; }
        this.done = true; this.cause = forced ? TRAPPED : out;
        return 0;
      }
      const d = this.absDir(a);
      const cell = this.target(d);
      const eat = cell === this.food;
      if (!eat) this.occ[this.seg(this.len - 1)] = 0;
      this.head = (this.head + 1) % this.A;
      this.body[this.head] = cell; this.occ[cell] = 1; this.dir = d;
      if (eat) {
        this.len++; this.score++; this.since = 0;
        if (this.len === this.A) { this.done = true; this.cause = WIN; this.food = -1; return 1; }
        this.placeFood();
        return 1;
      }
      this.since++;
      if (this.since > this.tlimit) { this.done = true; this.cause = TIMEOUT; }
      return 0;
    }
  }

  function floodAfterAction(g, a, stack, seen) {
    if (g.moveOutcome(a) !== 0) return [-1, 0];
    const G = g.G, A = g.A;
    const head = g.target(g.absDir(a));
    const eat = head === g.food;
    const L = g.len;
    const oldTail = g.seg(L - 1);
    const newTail = eat ? oldTail : g.seg(L - 2);
    for (let i = 0; i < A; i++) seen[i] = g.occ[i];
    if (!eat) seen[oldTail] = 0;
    seen[head] = 1;
    let sp = 0; stack[sp++] = head;
    let area = 0, tr = 0;
    while (sp > 0) {
      const cur = stack[--sp];
      const r = Math.floor(cur / G), c = cur % G;
      for (let k = 0; k < 4; k++) {
        const rr = r + DR[k], cc = c + DC[k];
        if (rr < 0 || rr >= G || cc < 0 || cc >= G) continue;
        const nb = rr * G + cc;
        if (nb === newTail && nb !== head) tr = 1;
        if (seen[nb] === 0) { seen[nb] = 1; area++; stack[sp++] = nb; }
      }
    }
    return [area, tr];
  }

  const N_FEATURES = 21;
  function computeFeatures(g, u, areas, scratch) {
    const G = g.G, A = g.A, L = g.len;
    const h = g.seg(0), hr = Math.floor(h / G), hc = h % G, d = g.dir;
    for (let i = 0; i < 5; i++) u[i] = 0;
    const f = g.food;
    if (f >= 0) {
      const vr = Math.floor(f / G) - hr, vc = (f % G) - hc;
      const fwd = vr * DR[d] + vc * DC[d];
      const dr = (d + 1) % 4;
      const lat = vr * DR[dr] + vc * DC[dr];
      u[0] = fwd > 0 ? 1 : 0; u[1] = fwd < 0 ? 1 : 0;
      u[2] = lat < 0 ? 1 : 0; u[3] = lat > 0 ? 1 : 0;
      u[4] = (Math.abs(vr) + Math.abs(vc)) / (2.0 * (G - 1));
    }
    const segidx = scratch.segidx;
    segidx.fill(-1);
    for (let i = 0; i < L; i++) segidx[g.seg(i)] = i;
    for (let a = 0; a < 3; a++) {
      const ad = (d + a - 1 + 4) % 4;
      let blocked = 0;
      for (let k = 1; k < 4; k++) {
        const rr = hr + DR[ad] * k, cc = hc + DC[ad] * k;
        if (rr < 0 || rr >= G || cc < 0 || cc >= G) blocked = 1;
        else { const si = segidx[rr * G + cc]; if (si >= 0 && si < L - k) blocked = 1; }
        if (k === 1) blocked = g.moveOutcome(a) !== 0 ? 1 : 0;
        u[5 + 3 * (k - 1) + a] = blocked;
      }
    }
    for (let a = 0; a < 3; a++) {
      const [area, tr] = floodAfterAction(g, a, scratch.stack, scratch.seen);
      areas[a] = area;
      if (area < 0) { u[14 + a] = 0; u[17 + a] = 0; }
      else { const ratio = area / L; u[14 + a] = ratio < 1 ? ratio : 1; u[17 + a] = tr; }
    }
    u[20] = L / A;
  }


  const N_FEATURES_V2 = 15, N_FEATURES_MAX = 21;
  function blockedAt(g, segidx, r, c, k) {
    const G = g.G;
    if (r < 0 || r >= G || c < 0 || c >= G) return true;
    const si = segidx[r * G + c];
    return si >= 0 && si < g.len - k;
  }
  function computeFeaturesV2(g, u, scratch) {
    const G = g.G, L = g.len;
    const h = g.seg(0), hr = Math.floor(h / G), hc = h % G, d = g.dir;
    for (let i = 0; i < N_FEATURES_V2; i++) u[i] = 0;
    const f = g.food;
    if (f >= 0) {
      const vr = Math.floor(f / G) - hr, vc = (f % G) - hc;
      const fwd = vr * DR[d] + vc * DC[d];
      const rd = (d + 1) % 4;
      const lat = vr * DR[rd] + vc * DC[rd];
      u[0] = fwd / (G - 1.0); u[1] = lat / (G - 1.0);
      u[2] = (Math.abs(vr) + Math.abs(vc)) / (2.0 * (G - 1));
      u[3] = fwd > 0 ? 1 : 0; u[4] = lat < 0 ? 1 : 0; u[5] = lat > 0 ? 1 : 0; u[6] = fwd < 0 ? 1 : 0;
    }
    const segidx = scratch.segidx;
    segidx.fill(-1);
    for (let i = 0; i < L; i++) segidx[g.seg(i)] = i;
    for (let a = 0; a < 3; a++) {
      u[7 + a] = g.moveOutcome(a) !== 0 ? 1 : 0;
      const ad = (d + a - 1 + 4) % 4;
      for (let k = 1; k <= G; k++) {
        if (blockedAt(g, segidx, hr + DR[ad] * k, hc + DC[ad] * k, k)) { u[10 + a] = 1.0 / k; break; }
      }
    }
    const ld = (d + 3) % 4, rd2 = (d + 1) % 4, fr = hr + DR[d], fc = hc + DC[d];
    u[13] = blockedAt(g, segidx, fr + DR[ld], fc + DC[ld], 1) ? 1 : 0;
    u[14] = blockedAt(g, segidx, fr + DR[rd2], fc + DC[rd2], 1) ? 1 : 0;
  }
  function features(g, u, areas, scratch, version) {
    if (version === 2) computeFeaturesV2(g, u, scratch); else computeFeatures(g, u, areas, scratch);
  }

  // ---- policies ----
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  class RateNet {
    // m: {n, indptr, indices, data, in_group, out_group, k_in, k_out, T, feat_idx, theta}
    constructor(m) {
      this.m = m;
      this.N = m.n; this.F = m.feat_idx.length;
      const F = this.F, ki = m.k_in, ko = m.k_out, th = m.theta;
      let p = 0;
      this.Win = th.slice(p, p + ki * F); p += ki * F;
      this.bin = th.slice(p, p + ki); p += ki;
      this.Wout = th.slice(p, p + 3 * ko); p += 3 * ko;
      this.bout = th.slice(p, p + 3); p += 3;
      this.alpha = 0.05 + 0.95 * sigmoid(th[p]);
      this.g = 2.0 * sigmoid(th[p + 1]);
      this.outSize = new Float64Array(Math.max(ko, 1));
      for (let i = 0; i < this.N; i++) if (m.out_group[i] >= 0) this.outSize[m.out_group[i]]++;
      this.mask = new Float64Array(this.N).fill(1);
      this.reset();
    }
    reset() {
      this.x = new Float64Array(this.N); this.phi = new Float64Array(this.N);
      this.rec = new Float64Array(this.N); this.drive = new Float64Array(Math.max(this.m.k_in, 1));
      this.rvec = new Float64Array(Math.max(this.m.k_out, 1)); this.logits = new Float64Array(3);
      this.usel = new Float64Array(this.F);
    }
    forward(u) {
      const m = this.m, N = this.N, F = this.F, ki = m.k_in, ko = m.k_out;
      for (let j = 0; j < F; j++) this.usel[j] = u[m.feat_idx[j]];
      for (let k = 0; k < ki; k++) {
        let s = this.bin[k];
        for (let j = 0; j < F; j++) s += this.Win[k * F + j] * this.usel[j];
        this.drive[k] = s;
      }
      const x = this.x, phi = this.phi, rec = this.rec, ip = m.indptr, ix = m.indices, wd = m.data;
      for (let t = 0; t < m.T; t++) {
        for (let i = 0; i < N; i++) {
          let s = 0.0;
          for (let p = ip[i]; p < ip[i + 1]; p++) s += wd[p] * phi[ix[p]];
          rec[i] = s;
        }
        for (let i = 0; i < N; i++) {
          let inp = 0.0;
          if (m.in_group[i] >= 0) inp = this.drive[m.in_group[i]];
          x[i] += this.alpha * (-x[i] + this.g * rec[i] + inp);
          phi[i] = Math.tanh(x[i]) * this.mask[i];
        }
      }
      this.rvec.fill(0);
      for (let i = 0; i < N; i++) if (m.out_group[i] >= 0) this.rvec[m.out_group[i]] += phi[i];
      for (let k = 0; k < ko; k++) this.rvec[k] /= Math.max(this.outSize[k], 1);
      for (let a = 0; a < 3; a++) {
        let s = this.bout[a];
        for (let k = 0; k < ko; k++) s += this.Wout[a * ko + k] * this.rvec[k];
        this.logits[a] = s;
      }
      let best = 0;
      for (let a = 1; a < 3; a++) if (this.logits[a] > this.logits[best]) best = a;
      return best;
    }
  }

  function playGame(net, seed, G, timeoutK, maxSteps) {
    const g = new Game(G, seed, timeoutK);
    const u = new Float64Array(N_FEATURES_MAX), areas = new Float64Array(3);
    const scratch = { stack: new Int32Array(G * G), seen: new Uint8Array(G * G), segidx: new Int32Array(G * G) };
    net.reset();
    const actions = [];
    while (!g.done && actions.length < (maxSteps || 1e9)) {
      features(g, u, areas, scratch, net.m.feat_version || 1);
      const a = net.forward(u);
      actions.push(a);
      g.step(a);
    }
    return { actions, score: g.score, cause: g.cause, steps: g.steps, len: g.len };
  }

  const api = { Game, RateNet, computeFeatures, computeFeaturesV2, features, N_FEATURES_MAX, floodAfterAction, playGame, rngSeed, rngNext,
    N_FEATURES, CAUSES, WIN };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FlySnake = api;
})(this);
