/* Smart Traffic Lights Using Deep Reinforcement Learning: project website.
 * Data comes from window.SITE_DATA (scripts in data/, written by tools/export_website_data.py).
 * Replays are recorded CityFlow runs; the decision model runs the trained PyTorch network in JavaScript. */
(function () {
  "use strict";
  const DATA = window.SITE_DATA || {};
  const IDX = DATA.index;
  const MODEL = DATA.model;
  const DIRS = ["W", "S", "E", "N"];
  // movement index of each lane (left, straight, right) on each approach
  const LINKS = { W: { s: 0, l: 1, r: 2 }, S: { s: 4, l: 5, r: 3 }, E: { s: 7, l: 8, r: 6 }, N: { s: 11, l: 9, r: 10 } };
  const DIR_NAME = { W: "West", S: "South", E: "East", N: "North" };
  const CTRL = {
    fixed: { name: "Fixed-time", color: "--fixed" },
    maxpressure: { name: "Max-pressure", color: "--mp" },
    dqn: { name: "DQN agent", color: "--dqn" },
    dqnplus: { name: "DQN+ agent", color: "--plus" },
  };
  const SCEN_NAME = {
    jinan_real: "Jinan, real traffic (training scenario)",
    jinan_2000: "Jinan, light traffic",
    jinan_2500: "Jinan, medium traffic",
    hangzhou_real: "Hangzhou, real traffic (unseen city)",
    hangzhou_5816: "Hangzhou, heavy traffic (unseen city)",
  };
  const CAR_M = 7.5;      // metres of road per stopped vehicle (5 m vehicle + 2.5 m gap)
  const STEP = 10;        // seconds between recorded frames
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const $ = (s, el = document) => el.querySelector(s);
  const fmtTime = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
  const lerp = (a, b, f) => a + (b - a) * f;
  const interName = (id) => { const m = id.match(/_(\d+)_(\d+)$/); return m ? `Intersection (${m[1]}, ${m[2]})` : id; };

  if (!IDX || !MODEL) {
    document.body.insertAdjacentHTML("afterbegin", '<p style="padding:16px;color:#d03b3b">Data files are missing (docs/data).</p>');
    return;
  }

  // ---------------------------------------------------------------- data
  const cache = {};
  function loadReplay(scen, ctrl) {
    const key = `${scen}__${ctrl}`;
    if (DATA[key]) return Promise.resolve(DATA[key]);
    if (cache[key]) return cache[key];
    const entry = IDX.replays.find((r) => r.scenario === scen && r.controller === ctrl);
    cache[key] = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = entry.file;
      s.onload = () => (DATA[key] ? resolve(DATA[key]) : reject(new Error("Invalid data file: " + entry.file)));
      s.onerror = () => reject(new Error("Could not load " + entry.file));
      document.head.appendChild(s);
    });
    return cache[key];
  }
  const netFor = (scen) => DATA[scen.startsWith("hangzhou") ? "net_hangzhou" : "net_jinan"];
  // lane[j][k] = 12 stopped + 12 vehicles (per movement) + 4 vehicles on the exit roads (W, S, E, N)
  const stopped = (rec, j, k, link) => rec.lane[j][k][link];
  const vehicles = (rec, j, k, link) => rec.lane[j][k][12 + link];
  const exitVeh = (rec, j, k, d) => rec.lane[j][k][24 + d];
  const apprSum = (rec, j, k, d, getter) => { const L = LINKS[DIRS[d]]; return getter(rec, j, k, L.l) + getter(rec, j, k, L.s) + getter(rec, j, k, L.r); };

  function phaseGreenDirs(phase) {
    const links = MODEL.phase_links[phase] || [];
    return DIRS.filter((d) => links.includes(LINKS[d].s) || links.includes(LINKS[d].l));
  }
  const phaseHas = (phase, link) => (MODEL.phase_links[phase] || []).includes(link);

  function fitCanvas(cv) {
    const r = cv.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  // ---------------------------------------------------------------- headline tiles
  const J = IDX.summary.find((s) => s.scenario === "jinan_real");
  $("#t-fixed").textContent = `${Math.round(J.fixed)} s`;
  $("#t-mp").textContent = `${Math.round(J.mp)} s`;
  $("#t-dqn").textContent = `${Math.round(J.dqn)} s`;
  $("#t-dqn-sub").textContent = `−${Math.round(100 * (J.fixed - J.dqn) / J.fixed)}% vs fixed-time · ` +
    `−${Math.round(100 * (J.mp - J.dqn) / J.mp)}% vs max-pressure`;

  // ================================================================ REPLAY
  const selScen = $("#sel-scen"), selL = $("#sel-left"), selR = $("#sel-right"), selInter = $("#sel-inter");
  const btnNet = $("#v-net"), btnClose = $("#v-close");
  Object.entries(SCEN_NAME).forEach(([k, v]) => selScen.add(new Option(v, k)));
  for (const sel of [selL, selR]) Object.entries(CTRL).forEach(([k, v]) => sel.add(new Option(v.name, k)));
  selScen.value = "jinan_real"; selL.value = "fixed"; selR.value = "dqn";

  const panels = [$("#panel-left"), $("#panel-right")].map((el) => ({
    el, cv: $("canvas", el), tip: $(".tip", el), rec: null, ctrl: null, geo: null, hover: null,
  }));
  const timeIn = $("#time"), clock = $("#clock"), btnPlay = $("#btn-play"), selSpeed = $("#sel-speed");
  let T = 0, playing = false, lastTs = 0, net = null, view = "network", selK = 0, flow = 0;

  function frameAt(rec, t) {
    const n = rec.total_wait.length;
    const j = Math.max(0, Math.min(Math.floor(t / STEP), n - 1));
    const j2 = Math.min(j + 1, n - 1);
    return { j, j2, f: Math.max(0, Math.min(1, (t - j * STEP) / STEP)) };
  }

  // Signal state at time t: the decision made at frame j is the phase stored at frame j+1;
  // if it differs from the previous phase, 5 s of amber come first.
  function signal(rec, k, t) {
    const { j, j2 } = frameAt(rec, t);
    if (j2 === j) return { phase: rec.phase[j][k], amber: false };
    if (rec.switched[j2][k] && t - j * STEP < 5) return { phase: rec.phase[j][k], amber: true };
    return { phase: rec.phase[j2][k], amber: false };
  }

  // ---------------------------------------------------------------- network view
  const MARGIN_M = 230;
  function netBounds(n) {
    const xs = n.intersections.map((i) => i.x), ys = n.intersections.map((i) => i.y);
    return { x0: Math.min(...xs) - MARGIN_M, x1: Math.max(...xs) + MARGIN_M, y0: Math.min(...ys) - MARGIN_M, y1: Math.max(...ys) + MARGIN_M };
  }
  function sizeCanvas(cv) {
    const w = cv.getBoundingClientRect().width || 400;
    let h;
    if (view === "close") h = Math.min(w, 560);
    else { const b = netBounds(net); h = Math.min(w * ((b.y1 - b.y0) / (b.x1 - b.x0)), window.innerHeight * 0.66); }
    cv.style.height = `${Math.round(h)}px`;
  }
  function buildGeo(n, w, h) {
    const { x0, x1, y0, y1 } = netBounds(n);
    const sc = Math.min(w / (x1 - x0), h / (y1 - y0));
    const ox = (w - (x1 - x0) * sc) / 2, oy = (h - (y1 - y0) * sc) / 2;
    return { sc, P: (x, y) => [ox + (x - x0) * sc, oy + (y1 - y) * sc], w, h };
  }

  function drawNetwork(p, ctx, w, h) {
    const g = (p.geo = buildGeo(net, w, h));
    const roadW = Math.max(10, 30 * g.sc), boxHalf = roadW * 0.6;
    const col = { road: css("--road"), q: css("--queue"), m: css("--moving"), box: css("--surface-2"),
      axis: css("--axis"), go: css("--go"), amber: css("--amber"), stop: css("--stop"), ink: css("--ink") };
    ctx.lineCap = "butt"; ctx.strokeStyle = col.road; ctx.lineWidth = roadW;
    for (const s of net.roads) {
      const a = g.P(s[0], s[1]), b = g.P(s[2], s[3]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    const { j, j2, f } = frameAt(p.rec, T);
    const laneOff = roadW * 0.25, qW = Math.max(4, roadW * 0.42), lightR = Math.max(3, Math.min(5, roadW * 0.22));
    net.intersections.forEach((it, k) => {
      const c = g.P(it.x, it.y);
      const sig = signal(p.rec, k, T), greens = phaseGreenDirs(sig.phase);
      for (let d = 0; d < 4; d++) {
        const ap = it.approach[DIRS[d]];
        const fp = g.P(ap.from[0], ap.from[1]);
        const len = Math.hypot(fp[0] - c[0], fp[1] - c[1]);
        const su = [(fp[0] - c[0]) / len, (fp[1] - c[1]) / len];
        const rn = [su[1], -su[0]];   // right-hand side of vehicles travelling towards the intersection
        const wq = lerp(apprSum(p.rec, j, k, d, stopped), apprSum(p.rec, j2, k, d, stopped), f);
        const vh = lerp(apprSum(p.rec, j, k, d, vehicles), apprSum(p.rec, j2, k, d, vehicles), f);
        const avail = Math.min(len, ap.length * g.sc) - boxHalf - 2;
        const qLen = Math.min(avail, wq * CAR_M * g.sc);
        const sx = c[0] + su[0] * boxHalf + rn[0] * laneOff, sy = c[1] + su[1] * boxHalf + rn[1] * laneOff;
        if (qLen > 0.5) {
          ctx.strokeStyle = col.q; ctx.lineWidth = qW; ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + su[0] * qLen, sy + su[1] * qLen); ctx.stroke();
        }
        const moving = Math.max(0, Math.round(vh - wq));
        if (moving > 0) {
          ctx.fillStyle = col.m;
          const span = avail - qLen - 3, n = Math.min(moving, Math.floor(span / 3));
          for (let i = 0; i < n; i++) {
            const dd = qLen + 3 + (i + 0.5) * (span / n);
            ctx.beginPath(); ctx.arc(sx + su[0] * dd, sy + su[1] * dd, 1.4, 0, 7); ctx.fill();
          }
        }
        const D = DIRS[d];
        ctx.fillStyle = sig.amber ? (greens.includes(D) ? col.amber : col.stop) : greens.includes(D) ? col.go : col.stop;
        const lx = c[0] + su[0] * (boxHalf + lightR) + rn[0] * (roadW / 2 + lightR + 1);
        const ly = c[1] + su[1] * (boxHalf + lightR) + rn[1] * (roadW / 2 + lightR + 1);
        ctx.beginPath(); ctx.arc(lx, ly, lightR, 0, 7); ctx.fill();
      }
      ctx.fillStyle = col.box; ctx.strokeStyle = col.axis; ctx.lineWidth = 1;
      ctx.fillRect(c[0] - boxHalf, c[1] - boxHalf, 2 * boxHalf, 2 * boxHalf);
      ctx.strokeRect(c[0] - boxHalf + 0.5, c[1] - boxHalf + 0.5, 2 * boxHalf - 1, 2 * boxHalf - 1);
      if (p.hover === k) {
        ctx.strokeStyle = col.ink; ctx.lineWidth = 2;
        ctx.strokeRect(c[0] - boxHalf - 3, c[1] - boxHalf - 3, 2 * boxHalf + 6, 2 * boxHalf + 6);
      }
    });
  }

  // ---------------------------------------------------------------- close-up view
  const SU = { W: [-1, 0], S: [0, 1], E: [1, 0], N: [0, -1] };   // screen direction from the centre to each approach
  function carRect(ctx, x, y, su, len, wid) {
    const horiz = su[1] === 0, w = horiz ? len : wid, h = horiz ? wid : len, r = Math.min(2, wid / 3);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - w / 2, y - h / 2, w, h, r); else ctx.rect(x - w / 2, y - h / 2, w, h);
    ctx.fill();
  }
  function laneArrow(ctx, x, y, su, kind, s) {
    // painted lane marking pointing towards the intersection: left, straight or right
    const v = [-su[0], -su[1]], left = [v[1], -v[0]];
    ctx.beginPath();
    const tail = [x - v[0] * s, y - v[1] * s];
    ctx.moveTo(tail[0], tail[1]); ctx.lineTo(x, y);
    let tip, dir;
    if (kind === "s") { tip = [x + v[0] * s * 0.6, y + v[1] * s * 0.6]; dir = v; }
    else { const sgn = kind === "l" ? 1 : -1; dir = [left[0] * sgn, left[1] * sgn]; tip = [x + dir[0] * s * 0.7, y + dir[1] * s * 0.7]; }
    ctx.lineTo(tip[0], tip[1]); ctx.stroke();
    const a = Math.atan2(dir[1], dir[0]), hs = s * 0.45;
    ctx.beginPath(); ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - hs * Math.cos(a - 0.5), tip[1] - hs * Math.sin(a - 0.5));
    ctx.lineTo(tip[0] - hs * Math.cos(a + 0.5), tip[1] - hs * Math.sin(a + 0.5)); ctx.closePath(); ctx.fill();
  }
  function movementArrow(ctx, c, bh, D, lane, off, lw) {
    const su = SU[D], rn = [su[1], -su[0]];
    const s0 = [c[0] + su[0] * bh + rn[0] * off, c[1] + su[1] * bh + rn[1] * off];
    let e, cp = null;
    if (lane === "s") e = [c[0] - su[0] * bh + rn[0] * off, c[1] - su[1] * bh + rn[1] * off];
    else {
      const lv = [-rn[0], -rn[1]], rn2 = [lv[1], -lv[0]];
      e = [c[0] + lv[0] * bh - rn2[0] * off, c[1] + lv[1] * bh - rn2[1] * off];
      cp = [c[0] + rn[0] * off * 0.3 + lv[0] * off * 0.3, c[1] + rn[1] * off * 0.3 + lv[1] * off * 0.3];
    }
    ctx.beginPath(); ctx.moveTo(s0[0], s0[1]);
    if (cp) ctx.quadraticCurveTo(cp[0], cp[1], e[0], e[1]); else ctx.lineTo(e[0], e[1]);
    ctx.stroke();
    const from = cp || s0, ang = Math.atan2(e[1] - from[1], e[0] - from[0]), ah = Math.max(6, lw);
    ctx.beginPath(); ctx.moveTo(e[0], e[1]);
    ctx.lineTo(e[0] - ah * Math.cos(ang - 0.45), e[1] - ah * Math.sin(ang - 0.45));
    ctx.lineTo(e[0] - ah * Math.cos(ang + 0.45), e[1] - ah * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fill();
  }

  function drawClose(p, ctx, w, h) {
    p.geo = null;
    const k = Math.min(selK, net.intersections.length - 1), it = net.intersections[k];
    const size = Math.min(w, h), c = [w / 2, h / 2];
    const lw = size * 0.027, bh = lw * 3.6, arm = size / 2 - bh;
    const carL = lw * 1.2, carW = lw * 0.62, gap = lw * 0.32, pitch = carL + gap, pxPerM = pitch / CAR_M;
    const col = { road: css("--road"), box: css("--surface-2"), mark: css("--axis"), q: css("--queue"), m: css("--moving"),
      go: css("--go"), amber: css("--amber"), stop: css("--stop"), ink2: css("--ink-2"), ink: css("--ink") };
    ctx.fillStyle = col.road;
    ctx.fillRect(0, c[1] - 3 * lw, w, 6 * lw); ctx.fillRect(c[0] - 3 * lw, 0, 6 * lw, h);
    ctx.fillStyle = col.box; ctx.fillRect(c[0] - bh, c[1] - bh, 2 * bh, 2 * bh);
    const { j, j2, f } = frameAt(p.rec, T);
    const sig = signal(p.rec, k, T);
    const labelFont = `600 ${Math.max(10, Math.round(size * 0.026))}px system-ui, sans-serif`;
    let totalStopped = 0;
    DIRS.forEach((D, d) => {
      const su = SU[D], rn = [su[1], -su[0]];
      const P = (dist, off) => [c[0] + su[0] * dist + rn[0] * off, c[1] + su[1] * dist + rn[1] * off];
      // centre line and lane dividers
      ctx.strokeStyle = col.mark; ctx.lineWidth = 1.5; ctx.setLineDash([]);
      let a = P(bh, 0), b = P(bh + arm + 20, 0);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.lineWidth = 1; ctx.setLineDash([lw * 0.8, lw * 0.8]);
      for (const off of [lw, 2 * lw, -lw, -2 * lw]) {
        a = P(bh, off); b = P(bh + arm + 20, off);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      ctx.setLineDash([]);
      const roadM = (it.approach[D] && it.approach[D].length) || 400;
      const visM = (arm - 6) / pxPerM;
      ["l", "s", "r"].forEach((lane, i) => {
        const link = LINKS[D][lane], off = (i + 0.5) * lw;
        // stop line coloured by this lane's signal (right turns are always permitted)
        let lc = col.go;
        if (lane !== "r") lc = sig.amber ? (phaseHas(sig.phase, link) ? col.amber : col.stop) : phaseHas(sig.phase, link) ? col.go : col.stop;
        ctx.strokeStyle = lc; ctx.lineWidth = Math.max(3, lw * 0.22);
        a = P(bh + 2, off - lw / 2 + 1); b = P(bh + 2, off + lw / 2 - 1);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        ctx.strokeStyle = col.mark; ctx.fillStyle = col.mark; ctx.lineWidth = 1.5;
        const ap = P(bh + lw * 1.25, off);
        laneArrow(ctx, ap[0], ap[1], su, lane, lw * 0.5);
        // stopped vehicles, from the stop line outwards
        const nq = lerp(stopped(p.rec, j, k, link), stopped(p.rec, j2, k, link), f);
        totalStopped += nq;
        const start = bh + lw * 0.45;
        const maxCars = Math.floor((arm - lw * 0.45 - lw * 2) / pitch);
        const full = Math.min(Math.floor(nq), maxCars);
        ctx.fillStyle = col.q;
        for (let q = 0; q < full; q++) { const pt = P(start + q * pitch + carL / 2, off); carRect(ctx, pt[0], pt[1], su, carL, carW); }
        if (full < maxCars && nq - full > 0.05) {
          ctx.globalAlpha = nq - full; const pt = P(start + full * pitch + carL / 2, off);
          carRect(ctx, pt[0], pt[1], su, carL, carW); ctx.globalAlpha = 1;
        }
        if (Math.floor(nq) > maxCars) {
          ctx.fillStyle = col.ink2; ctx.font = labelFont; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          const pt = P(bh + arm - lw * 0.8, off); ctx.fillText(`+${Math.floor(nq) - maxCars}`, pt[0], pt[1]);
        }
        // moving vehicles on the visible part of the road, evenly spaced and drifting towards the stop line
        const nm = Math.max(0, lerp(vehicles(p.rec, j, k, link), vehicles(p.rec, j2, k, link), f) - nq);
        const qEnd = start + Math.min(nq, maxCars) * pitch + 2;
        const room = bh + arm - 4 - qEnd;
        if (room > pitch && nm > 0.5) {
          const share = Math.min(1, (room / pxPerM) / Math.max(1, roadM - nq * CAR_M));
          const n = Math.min(Math.round(nm * share), Math.floor(room / pitch));
          if (n > 0) {
            const sp = room / n, shift = flow % sp;
            ctx.fillStyle = col.m;
            for (let q = 0; q < n; q++) {
              const dist = qEnd + carL / 2 + ((q * sp - shift + room) % room);
              if (dist > bh + arm - 2) continue;
              const pt = P(dist, off); carRect(ctx, pt[0], pt[1], su, carL, carW);
            }
          }
        }
      });
      // vehicles leaving on the exit road, drifting outwards
      const nx = lerp(exitVeh(p.rec, j, k, d), exitVeh(p.rec, j2, k, d), f);
      const nShow = Math.min(Math.round(nx * Math.min(1, visM / roadM)), 3 * Math.floor(arm / pitch / 2));
      if (nShow > 0) {
        ctx.fillStyle = col.m;
        const per = Math.ceil(nShow / 3), sp = (arm - 6) / Math.max(per, 1);
        for (let q = 0; q < nShow; q++) {
          const lane = q % 3, idx = Math.floor(q / 3);
          const dist = bh + 4 + carL / 2 + ((idx * sp + flow + (lane * sp) / 3) % (arm - 6 - carL));
          const pt = P(dist, -(lane + 0.5) * lw); carRect(ctx, pt[0], pt[1], su, carL, carW);
        }
      }
      // approach label beside the road
      const ns = Math.round(lerp(apprSum(p.rec, j, k, d, stopped), apprSum(p.rec, j2, k, d, stopped), f));
      ctx.fillStyle = col.ink2; ctx.font = labelFont; ctx.textBaseline = "middle";
      const lp = P(bh + arm * 0.7, 3 * lw + lw * 1.1);
      if (su[1] === 0) { ctx.textAlign = "center"; ctx.fillText(`${DIR_NAME[D]}: ${ns} stopped`, lp[0], lp[1]); }
      else { ctx.textAlign = rn[0] > 0 ? "left" : "right"; ctx.fillText(`${DIR_NAME[D]}: ${ns} stopped`, lp[0], lp[1]); }
    });
    // movements currently given green
    if (!sig.amber) {
      ctx.strokeStyle = col.go; ctx.fillStyle = col.go; ctx.lineWidth = Math.max(2, lw * 0.16); ctx.globalAlpha = 0.9;
      for (const link of MODEL.phase_links[sig.phase] || []) {
        const D = DIRS.find((x) => LINKS[x].s === link || LINKS[x].l === link);
        const lane = LINKS[D].s === link ? "s" : "l";
        movementArrow(ctx, c, bh, D, lane, ((lane === "l" ? 0 : 1) + 0.5) * lw, lw * 0.5);
      }
      ctx.globalAlpha = 1;
    }
    const cap = $(".closecap", p.el);
    cap.innerHTML = `<b>${interName(it.id)}</b><span>${sig.amber ? "Phase change: amber" : `Green: ${MODEL.phase_names[sig.phase]}`}</span>` +
      `<span>${Math.round(totalStopped)} vehicles stopped here</span>`;
  }

  function drawPanel(p) {
    if (net) sizeCanvas(p.cv);
    const { ctx, w, h } = fitCanvas(p.cv);
    ctx.clearRect(0, 0, w, h);
    if (!p.rec || !net || p.rec.intersections.length !== net.intersections.length) return;
    $(".closecap", p.el).hidden = view !== "close";
    if (view === "close") drawClose(p, ctx, w, h); else drawNetwork(p, ctx, w, h);
    const s = p.rec, { j, j2, f } = frameAt(s, T);
    $(".s-wait", p.el).textContent = Math.round(lerp(s.total_wait[j], s.total_wait[j2], f)).toLocaleString();
    $(".s-fin", p.el).textContent = Math.round(lerp(s.finished[j], s.finished[j2], f)).toLocaleString();
    $(".s-tt", p.el).textContent = T < 60 ? "–" : `${Math.round(lerp(s.avg_travel[j], s.avg_travel[j2], f))} s`;
    if (view === "network" && p.hover !== null) showMapTip(p); else p.tip.hidden = true;
  }

  function showMapTip(p) {
    const k = p.hover, it = net.intersections[k];
    const { j2 } = frameAt(p.rec, T);
    const c = p.geo.P(it.x, it.y);
    const sig = signal(p.rec, k, T);
    const ph = sig.amber ? "phase change (amber)" : MODEL.phase_names[sig.phase];
    const ws = [0, 1, 2, 3].map((d) => apprSum(p.rec, j2, k, d, stopped));
    p.tip.innerHTML = `<b>${interName(it.id)}</b><br>Green: ${ph}<br>Stopped: W ${ws[0]} · S ${ws[1]} · E ${ws[2]} · N ${ws[3]}` +
      `<br><span class="tip-hint">Select for close-up</span>`;
    p.tip.hidden = false;
    const tw = p.tip.offsetWidth;
    let x = c[0] + 14; if (x + tw > p.geo.w) x = c[0] - tw - 14;
    p.tip.style.left = `${Math.max(4, x)}px`; p.tip.style.top = `${Math.max(4, c[1] - 30)}px`;
  }

  function nearestInter(p, e) {
    if (!p.geo || !net) return null;
    const r = p.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 26;
    net.intersections.forEach((it, k) => {
      const c = p.geo.P(it.x, it.y), d = Math.hypot(c[0] - x, c[1] - y);
      if (d < bd) { bd = d; best = k; }
    });
    return best;
  }
  panels.forEach((p) => {
    p.cv.addEventListener("pointermove", (e) => {
      if (view !== "network") return;
      const k = nearestInter(p, e);
      p.cv.style.cursor = k === null ? "default" : "pointer";
      if (k !== p.hover) { p.hover = k; drawPanel(p); }
    });
    p.cv.addEventListener("pointerleave", () => { p.hover = null; p.tip.hidden = true; if (view === "network") drawPanel(p); });
    p.cv.addEventListener("click", (e) => {
      if (view !== "network") return;
      const k = nearestInter(p, e);
      if (k !== null) { selK = k; selInter.value = k; setView("close"); }
    });
  });

  function setView(v) {
    view = v;
    btnNet.setAttribute("aria-selected", String(v === "network"));
    btnClose.setAttribute("aria-selected", String(v === "close"));
    $("#legend-net").hidden = v !== "network";
    $("#legend-close").hidden = v !== "close";
    panels.forEach((p) => { p.hover = null; p.tip.hidden = true; p.cv.style.cursor = "default"; });
    drawAll();
  }
  btnNet.addEventListener("click", () => setView("network"));
  btnClose.addEventListener("click", () => setView("close"));
  selInter.addEventListener("change", () => { selK = Number(selInter.value); if (view !== "close") setView("close"); else drawAll(); });

  function fillInterSelect() {
    selInter.innerHTML = "";
    net.intersections.forEach((it, k) => selInter.add(new Option(interName(it.id), k)));
    if (selK >= net.intersections.length) selK = 0;
    selInter.value = selK;
  }

  // ---------------------------------------------------------------- waiting chart
  const wc = $("#wait-chart"), wtip = $("#wait-tip");
  let chartHover = null, chartGeo = null;
  function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p; return v; }
  function drawChart() {
    const { ctx, w, h } = fitCanvas(wc);
    ctx.clearRect(0, 0, w, h);
    if (!panels.some((p) => p.rec)) return;
    const ml = 44, mr = 14, mt = 10, mb = 26;
    const n = panels.find((p) => p.rec).rec.total_wait.length;
    const ymax = niceMax(Math.max(10, ...panels.filter((p) => p.rec).flatMap((p) => p.rec.total_wait)));
    const X = (i) => ml + (i / (n - 1)) * (w - ml - mr), Y = (v) => mt + (1 - v / ymax) * (h - mt - mb);
    chartGeo = { n, ml, mr, w };
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const v = (ymax * i) / 4, y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(w - mr, y); ctx.stroke();
      ctx.fillText(Math.round(v).toLocaleString(), ml - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let m = 0; m <= 60; m += 10) ctx.fillText(`${m} min`, X((m * 60) / STEP), h - mb + 8);
    panels.forEach((p) => {
      if (!p.rec) return;
      ctx.strokeStyle = css(CTRL[p.ctrl].color); ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath();
      p.rec.total_wait.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
      ctx.stroke();
    });
    const px = X(T / STEP);
    ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, mt); ctx.lineTo(px, h - mb); ctx.stroke(); ctx.setLineDash([]);
    if (chartHover !== null) {
      const i = chartHover, x = X(i);
      ctx.strokeStyle = css("--axis"); ctx.beginPath(); ctx.moveTo(x, mt); ctx.lineTo(x, h - mb); ctx.stroke();
      let html = `<b>${fmtTime(i * STEP)}</b>`;
      panels.forEach((p) => {
        if (!p.rec) return;
        const v = p.rec.total_wait[i];
        ctx.fillStyle = css(CTRL[p.ctrl].color); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, Y(v), 4, 0, 7); ctx.fill(); ctx.stroke();
        html += `<div class="row"><i class="sw" style="--c:var(${CTRL[p.ctrl].color})"></i>${CTRL[p.ctrl].name}: <b>${v.toLocaleString()}</b> waiting</div>`;
      });
      wtip.innerHTML = html; wtip.hidden = false;
      const tw = wtip.offsetWidth;
      wtip.style.left = `${x + 12 + tw > w ? x - tw - 12 : x + 12}px`; wtip.style.top = "8px";
    } else wtip.hidden = true;
  }
  function chartIndex(e) {
    if (!chartGeo) return null;
    const r = wc.getBoundingClientRect(), x = e.clientX - r.left;
    const i = Math.round(((x - chartGeo.ml) / (chartGeo.w - chartGeo.ml - chartGeo.mr)) * (chartGeo.n - 1));
    return Math.max(0, Math.min(chartGeo.n - 1, i));
  }
  wc.addEventListener("pointermove", (e) => { chartHover = chartIndex(e); drawChart(); });
  wc.addEventListener("pointerleave", () => { chartHover = null; drawChart(); });
  wc.addEventListener("click", (e) => { const i = chartIndex(e); if (i !== null) seek(i * STEP); });

  // ---------------------------------------------------------------- player
  function drawAll() { panels.forEach(drawPanel); drawChart(); timeIn.value = T; clock.textContent = fmtTime(T); }
  function seek(t) { T = Math.max(0, Math.min(3600, t)); drawAll(); }
  function setPlaying(on) {
    playing = on; btnPlay.textContent = on ? "Pause" : "Play"; btnPlay.setAttribute("aria-label", on ? "Pause" : "Play");
    if (on) { if (T >= 3600) T = 0; lastTs = performance.now(); requestAnimationFrame(tick); }
  }
  function tick(ts) {
    if (!playing) return;
    const dt = Math.max(0, ts - lastTs) / 1000; lastTs = ts;
    T = Math.max(0, T + dt * Number(selSpeed.value));
    flow += dt * 40;   // drift of moving vehicles in the close-up, pixels per second
    if (T >= 3600) { T = 3600; setPlaying(false); }
    drawAll();
    if (playing) requestAnimationFrame(tick);
  }
  btnPlay.addEventListener("click", () => setPlaying(!playing));
  timeIn.addEventListener("input", () => seek(Number(timeIn.value)));

  let loadToken = 0;
  async function loadSelection() {
    const my = ++loadToken;
    const scen = selScen.value, ctrls = [selL.value, selR.value];
    try {
      const recs = await Promise.all(ctrls.map((c) => loadReplay(scen, c)));
      if (my !== loadToken) return;
      const newNet = netFor(scen), cityChanged = newNet !== net;
      net = newNet;
      if (cityChanged) fillInterSelect();
      panels.forEach((p, i) => {
        p.rec = recs[i]; p.ctrl = ctrls[i]; p.hover = null; p.tip.hidden = true;
        $(".pname", p.el).innerHTML = `<i class="sw" style="--c:var(${CTRL[p.ctrl].color})"></i>${CTRL[p.ctrl].name}`;
        $(".pfinal", p.el).textContent = `Hourly mean ${p.rec.final.avg_travel_time} s` +
          (p.ctrl === "fixed" && p.rec.controller.includes(", ") ? ` · ${p.rec.controller.split(", ")[1]}` : "");
      });
      $("#wait-legend").innerHTML = panels.map((p) => `<span><i style="--c:var(${CTRL[p.ctrl].color})"></i>${CTRL[p.ctrl].name}</span>`).join("");
      drawAll();
    } catch (err) {
      panels[0].el.insertAdjacentHTML("beforebegin", `<p class="fine" style="color:var(--stop)">${err.message}</p>`);
    }
  }
  [selScen, selL, selR].forEach((s) => s.addEventListener("change", loadSelection));
  loadSelection();

  // ================================================================ DECISION MODEL
  const L = MODEL.layers;
  function qValues(veh, wait, phase) {
    const x = new Float64Array(32);
    for (let i = 0; i < 12; i++) { x[i] = veh[i] / MODEL.count_scale; x[12 + i] = wait[i] / MODEL.count_scale; }
    x[24 + phase] = 1;
    let h = x;
    L.forEach((layer, li) => {
      const out = new Float64Array(layer.b.length);
      for (let o = 0; o < out.length; o++) {
        let s = layer.b[o];
        const row = layer.W[o];
        for (let i = 0; i < row.length; i++) s += row[i] * h[i];
        out[o] = li < L.length - 1 ? Math.max(0, s) : s;
      }
      h = out;
    });
    return Array.from(h);
  }
  const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

  (function selfTest() {
    let worst = 0, same = 0;
    for (const t of MODEL.tests) {
      const q = qValues(t.veh, t.wait, t.phase);
      worst = Math.max(worst, ...q.map((v, i) => Math.abs(v - t.q[i])));
      if (argmax(q) === argmax(t.q)) same++;
    }
    const el = $("#selftest"), ok = worst < 2e-3 && same === MODEL.tests.length;
    el.className = "selftest " + (ok ? "ok" : "bad");
    el.textContent = ok
      ? `Verified against PyTorch: identical decisions on all ${MODEL.tests.length} reference cases (largest score difference ${worst.toFixed(4)}).`
      : `Verification failed: largest score difference ${worst.toFixed(4)}, matching decisions ${same} of ${MODEL.tests.length}.`;
  })();

  const st = { wait: new Array(12).fill(0), mov: new Array(12).fill(0), phase: 0 };
  let recorded = null, edited = false;
  const selPhase = $("#sel-phase");
  MODEL.phase_names.forEach((n, i) => selPhase.add(new Option(n, i)));
  const sliderBox = $("#sliders");
  const sliders = {};
  DIRS.forEach((d) => {
    const box = document.createElement("div");
    box.className = "appr";
    box.innerHTML = `<h4>${DIR_NAME[d]} approach</h4>` +
      [["ws", "Stopped, straight", 40], ["wl", "Stopped, left turn", 40], ["mv", "Moving towards the signal", 60]]
        .map(([k, lab, max]) => `<label class="sl"><span>${lab}</span><output></output>` +
          `<input type="range" min="0" max="${max}" step="1" value="0" data-d="${d}" data-k="${k}" aria-label="${DIR_NAME[d]} approach: ${lab}"></label>`).join("");
    sliderBox.appendChild(box);
    sliders[d] = {};
    box.querySelectorAll("input").forEach((inp) => { sliders[d][inp.dataset.k] = inp; });
  });

  function syncSliders() {
    DIRS.forEach((d) => {
      const { s, l } = LINKS[d];
      const vals = { ws: st.wait[s], wl: st.wait[l], mv: st.mov[s] + st.mov[l] };
      for (const k of ["ws", "wl", "mv"]) {
        const inp = sliders[d][k];
        inp.value = Math.min(Number(inp.max), vals[k]);
        inp.previousElementSibling.textContent = vals[k];
      }
    });
    selPhase.value = st.phase;
  }
  function setInfo(html) { $("#moment-info").innerHTML = html; }
  function markEdited() { if (!edited) { edited = true; setInfo("Custom situation."); } }
  sliderBox.addEventListener("input", (e) => {
    const inp = e.target; if (!inp.dataset.d) return;
    const d = inp.dataset.d, k = inp.dataset.k, v = Number(inp.value), { s, l } = LINKS[d];
    if (k === "ws") st.wait[s] = v;
    if (k === "wl") st.wait[l] = v;
    if (k === "mv") { st.mov[l] = Math.min(st.mov[l], v); st.mov[s] = v - st.mov[l]; }
    inp.previousElementSibling.textContent = v;
    markEdited(); updateAsk();
  });
  selPhase.addEventListener("change", () => { st.phase = Number(selPhase.value); markEdited(); updateAsk(); });

  $("#btn-clear").addEventListener("click", () => {
    st.wait.fill(0); st.mov.fill(0); st.phase = 0; recorded = null; edited = true;
    setInfo("Empty intersection.");
    syncSliders(); updateAsk();
  });
  $("#btn-moment").addEventListener("click", async () => {
    const rec = await loadReplay("jinan_real", "dqn");
    const j = 6 + Math.floor(Math.random() * (rec.lane.length - 14));
    const k = Math.floor(Math.random() * rec.lane[j].length);
    const o = rec.lane[j][k];
    for (let i = 0; i < 12; i++) { st.wait[i] = o[i]; st.mov[i] = o[12 + i] - o[i]; }
    st.phase = rec.phase[j][k];
    recorded = rec.phase[j + 1][k];
    edited = false;
    setInfo(`Recorded situation: Jinan real traffic, ${fmtTime(j * STEP)} into the hour, ${interName(rec.intersections[k])}. ` +
      `Decision taken in the simulation: <b>${MODEL.phase_names[recorded]}</b>.`);
    syncSliders(); updateAsk();
  });

  function updateAsk() {
    const veh = st.wait.map((w, i) => w + st.mov[i]);
    const q = qValues(veh, st.wait, st.phase);
    const best = argmax(q);
    const sorted = [...q].sort((a, b) => b - a);
    const lo = Math.min(...q), hi = Math.max(...q), span = Math.max(hi - lo, 0.05);
    $("#qlist").innerHTML = q.map((v, i) => {
      const pos = 8 + ((v - lo) / span) * 84;
      return `<div class="qrow${i === best ? " best" : ""}" role="listitem"><span class="qn">${MODEL.phase_names[i]}</span>` +
        `<span class="qtrack"><span class="qdot" style="left:${pos}%"></span></span><span class="qv">${v.toFixed(2)}</span></div>`;
    }).join("");
    const greens = phaseGreenDirs(best).map((d) => DIR_NAME[d].toLowerCase());
    const serves = greens.length > 1 ? `${greens[0]} and ${greens[1]} approaches` : `${greens[0]} approach`;
    let html = `<b>${MODEL.phase_names[best]}</b><span class="dec-detail">Serves the ${serves} · ` +
      `margin over the next phase ${(sorted[0] - sorted[1]).toFixed(2)} · ` +
      `${best === st.phase ? "keeps the current green" : "phase change, preceded by 5 s amber"}</span>`;
    if (recorded !== null && !edited) {
      html += recorded === best
        ? `<span class="dec-match ok">Matches the decision recorded in the simulation.</span>`
        : `<span class="dec-match bad">Differs from the recorded decision (${MODEL.phase_names[recorded]}).</span>`;
    }
    $("#decision").innerHTML = html;
    drawAsk(best);
  }

  // ---------------------------------------------------------------- decision diagram
  const askCv = $("#ask-canvas");
  function drawAsk(best) {
    const { ctx, w } = fitCanvas(askCv);
    const size = w, c = [size / 2, size / 2], rw = size * 0.3, bh = rw / 2;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = css("--road");
    ctx.fillRect(0, c[1] - bh, size, rw); ctx.fillRect(c[0] - bh, 0, rw, size);
    ctx.fillStyle = css("--surface-2"); ctx.fillRect(c[0] - bh, c[1] - bh, rw, rw);
    ctx.strokeStyle = css("--axis"); ctx.setLineDash([6, 6]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, c[1]); ctx.lineTo(c[0] - bh, c[1]); ctx.moveTo(c[0] + bh, c[1]); ctx.lineTo(size, c[1]);
    ctx.moveTo(c[0], 0); ctx.lineTo(c[0], c[1] - bh); ctx.moveTo(c[0], c[1] + bh); ctx.lineTo(c[0], size); ctx.stroke();
    ctx.setLineDash([]);
    const laneOf = { l: 0.09, s: 0.25, r: 0.41 };
    const carL = Math.max(6, size * 0.028), carW = Math.max(4, rw * 0.11), gap = 2;
    ctx.font = `600 ${Math.round(size * 0.04)}px system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    DIRS.forEach((d) => {
      const su = SU[d], rn = [su[1], -su[0]];
      const room = size / 2 - bh - 6, maxCars = Math.floor(room / (carL + gap));
      for (const lane of ["l", "s", "r"]) {
        const link = LINKS[d][lane], off = laneOf[lane] * rw;
        const nWait = st.wait[link], nMov = st.mov[link];
        const sx = c[0] + su[0] * (bh + 3) + rn[0] * off, sy = c[1] + su[1] * (bh + 3) + rn[1] * off;
        ctx.fillStyle = css("--queue");
        const shown = Math.min(nWait, maxCars);
        for (let i = 0; i < shown; i++) {
          const a = i * (carL + gap) + carL / 2;
          carRect(ctx, sx + su[0] * a, sy + su[1] * a, su, carL, carW);
        }
        ctx.fillStyle = css("--moving");
        const free = maxCars - shown, nm = Math.min(nMov, free);
        for (let i = 0; i < nm; i++) {
          const a = shown * (carL + gap) + (i + 0.5) * ((free * (carL + gap)) / Math.max(nm, 1));
          carRect(ctx, sx + su[0] * a, sy + su[1] * a, su, carL * 0.85, carW * 0.85);
        }
        if (nWait > maxCars) {
          ctx.fillStyle = css("--ink-2");
          ctx.fillText(`+${nWait - maxCars}`, sx + su[0] * (room - 6), sy + su[1] * (room - 6));
        }
      }
      ctx.fillStyle = css(phaseGreenDirs(best).includes(d) ? "--go" : "--stop");
      const lx = c[0] + su[0] * (bh + 8) + rn[0] * (bh + 9), ly = c[1] + su[1] * (bh + 8) + rn[1] * (bh + 9);
      ctx.beginPath(); ctx.arc(lx, ly, Math.max(5, size * 0.016), 0, 7); ctx.fill();
      ctx.fillStyle = css("--ink-2");
      const tx = c[0] + su[0] * (size / 2 - 12) - rn[0] * (bh + 14), ty = c[1] + su[1] * (size / 2 - 12) - rn[1] * (bh + 14);
      ctx.fillText(d, tx, ty);
    });
    ctx.strokeStyle = css("--go"); ctx.fillStyle = css("--go"); ctx.lineWidth = Math.max(2.5, size * 0.009);
    for (const link of MODEL.phase_links[best]) {
      const d = DIRS.find((x) => LINKS[x].s === link || LINKS[x].l === link);
      const lane = LINKS[d].s === link ? "s" : "l";
      movementArrow(ctx, c, bh, d, lane, laneOf[lane] * rw, Math.max(7, size * 0.025));
    }
  }

  // initial example: long queue on the north approach
  st.wait[LINKS.N.s] = 9; st.wait[LINKS.N.l] = 3; st.wait[LINKS.W.s] = 4; st.mov[LINKS.E.s] = 2; st.phase = 6;
  setInfo("Example situation: a long queue on the north approach.");
  syncSliders(); updateAsk();

  // ================================================================ RESULTS TABLE
  (function results() {
    const rows = IDX.summary;
    const th = `<thead><tr><th>Traffic scenario</th><th><i class="sw" style="--c:var(--fixed)"></i>Fixed-time</th>` +
      `<th><i class="sw" style="--c:var(--mp)"></i>Max-pressure</th><th><i class="sw" style="--c:var(--dqn)"></i>DQN</th>` +
      `<th><i class="sw" style="--c:var(--plus)"></i>DQN+</th><th>DQN vs fixed-time</th><th>DQN vs max-pressure</th></tr></thead>`;
    const body = rows.map((r) => {
      const m = Math.min(r.fixed, r.mp, r.dqn, r.plus);
      const cell = (v, extra = "") => `<td class="${v === m ? "win" : ""}">${v.toFixed(1)}${extra}</td>`;
      return `<tr><td>${SCEN_NAME[r.scenario]}</td>${cell(r.fixed, ` <span class="rng">(${r.fixed_green} s green)</span>`)}` +
        cell(r.mp) + cell(r.dqn, ` <span class="rng">(${Math.round(r.dqn_min)}–${Math.round(r.dqn_max)})</span>`) +
        cell(r.plus, ` <span class="rng">(${Math.round(r.plus_min)}–${Math.round(r.plus_max)})</span>`) +
        `<td>−${(100 * (r.fixed - r.dqn) / r.fixed).toFixed(1)}%</td><td>−${(100 * (r.mp - r.dqn) / r.mp).toFixed(1)}%</td></tr>`;
    }).join("");
    $("#res-table").innerHTML = th + `<tbody>${body}</tbody>`;
    const hh = rows.find((r) => r.scenario === "hangzhou_5816");
    $("#res-note").textContent = `Hangzhou heavy traffic is the most difficult case. The three DQN agents ranged from ` +
      `${Math.round(hh.dqn_min)} to ${Math.round(hh.dqn_max)} s, and one was slower than max-pressure (${Math.round(hh.mp)} s). ` +
      `DQN+ narrowed the range to ${Math.round(hh.plus_min)}–${Math.round(hh.plus_max)} s, with all three agents faster than ` +
      `max-pressure. The replays above use one trained agent of each type.`;
  })();

  // ---------------------------------------------------------------- redraw on resize / theme change
  let raf = 0;
  const redraw = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { drawAll(); updateAsk(); }); };
  window.addEventListener("resize", redraw);
  if (window.matchMedia) matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
})();
