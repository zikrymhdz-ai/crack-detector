/* CrackScan mobile client
 *
 * Talks to the Flask server in this repository:
 *   GET  /api/mobile/health            handshake for the connect screen
 *   POST /api/mobile/scan              analyse + store a photo   (multipart or JSON base64)
 *   POST /analyze_live                 analyse a preview frame   (JSON base64, not stored)
 *   GET  /api/cracks?limit=&images=0   inspection history
 *   PATCH /api/cracks/<id>             inspection status
 *   DELETE /api/mobile/records/<id>    remove a scan
 *
 * No framework, no build step: Capacitor serves this folder straight into the
 * WebView, so everything here is plain ES2019.
 */
(function () {
  'use strict';

  // ── tiny DOM helpers ─────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root) => Array.prototype.slice.call((root || document).querySelectorAll(sel));

  const LEVEL_COLOR = {
    0: '#00d16f',
    1: '#ffc700',
    2: '#ff8a00',
    3: '#ff3b30'
  };
  const LEVEL_NAME = { 0: 'SAFE', 1: 'MINOR', 2: 'MEDIUM', 3: 'SEVERE' };
  const SEV_NAME = { safe: 'SAFE', minor: 'MINOR', moderate: 'MEDIUM', severe: 'SEVERE' };

  // ── state ────────────────────────────────────────────────────────
  const LS_SERVER = 'crackscan.server';
  const state = {
    server: '',
    connected: false,
    health: null,
    platform: 'web',
    live: { on: false, busy: false, stream: null, timer: null, last: null },
    frame: { w: 0, h: 0 },
    current: null,
    filter: ''
  };

  const isNative = !!(window.Capacitor && window.Capacitor.isNativePlatform && window.Capacitor.isNativePlatform());

  // ── misc ui ──────────────────────────────────────────────────────
  let toastTimer = null;
  function toast(message, bad) {
    const el = $('toast');
    el.textContent = message;
    el.classList.toggle('is-bad', !!bad);
    el.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('is-on'), 2600);
  }

  function normalizeServer(raw) {
    let s = (raw || '').trim();
    if (!s) return '';
    if (!/^https?:\/\//i.test(s)) s = 'http://' + s;
    s = s.replace(/\/+$/, '');
    return s;
  }

  function defaultServer() {
    // The mobile bundle inside Capacitor is served from https://localhost with no
    // port; anything else (desktop browser on the Flask app, phone browser hitting
    // /app/, or a LAN preview) is the Flask server itself, so default to it.
    const httpish = location.protocol === 'http:' || location.protocol === 'https:';
    if (!httpish) return '';
    if (location.hostname === 'localhost' && !location.port) return '';
    return location.origin;
  }

  function setConn(ok, text) {
    state.connected = !!ok;
    const chip = $('btnReconnect');
    chip.classList.toggle('is-on', !!ok);
    chip.classList.toggle('is-off', !ok && !!text);
    $('connChipText').textContent = text || (ok ? 'Online' : 'Offline');
    $('connLine').textContent = ok
      ? (state.server.replace(/^https?:\/\//, '') + ' · ' + (state.health && state.health.model || 'server'))
      : 'Tap to connect to your PC';
  }

  // ── api ──────────────────────────────────────────────────────────
  async function api(path, options, timeoutMs) {
    if (!state.server) throw new Error('No server configured');
    const opts = Object.assign({}, options || {});
    opts.cache = 'no-store';
    const ctrl = new AbortController();
    opts.signal = ctrl.signal;
    const timer = setTimeout(() => ctrl.abort(), timeoutMs || 20000);
    let res;
    try {
      res = await fetch(state.server + path, opts);
    } catch (e) {
      clearTimeout(timer);
      const err = new Error(e.name === 'AbortError' ? 'Server timed out' : 'Cannot reach the server');
      err.offline = true;
      throw err;
    }
    clearTimeout(timer);
    let body = null;
    const text = await res.text();
    try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
    if (!res.ok) {
      const msg = (body && (body.error || body.message)) || ('Server error ' + res.status);
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return body;
  }

  async function connect(quiet) {
    if (!state.server) {
      setConn(false, '');
      return false;
    }
    try {
      const h = await api('/api/mobile/health', { method: 'GET' }, 8000);
      state.health = h;
      setConn(true, 'Online');
      $('kvHost').textContent = h.host || state.server;
      $('kvVersion').textContent = 'v' + (h.version || '?');
      $('kvModel').textContent = h.model || '—';
      $('kvMasonry').textContent = h.masonry ? 'Enabled' : 'Unavailable';
      loadSettings();
      if (!quiet) toast('Connected to ' + (h.host || state.server));
      return true;
    } catch (e) {
      state.health = null;
      setConn(false, '');
      loadSettings();
      if (!quiet) toast(e.message || 'Connection failed', true);
      return false;
    }
  }

  // ── tabs / navigation ────────────────────────────────────────────
  function showScreen(name) {
    $$('.screen').forEach((s) => s.classList.toggle('is-active', s.id === 'screen-' + name));
    $$('#tabbar .tab').forEach((t) => t.classList.toggle('is-on', t.dataset.screen === name));
    if (name !== 'scan' && state.live.on) stopLive();
    if (name === 'history') loadHistory();
  }

  // ── camera ───────────────────────────────────────────────────────
  async function enableCamera() {
    if (state.live.stream) {
      // already running (e.g. coming back from another tab)
      $('viewport').classList.add('is-live');
      requestAnimationFrame(() => { sizeOverlay(); setLive(true); });
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast('This device has no camera bridge', true);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      state.live.stream = stream;
      const video = $('video');
      video.srcObject = stream;
      await video.play().catch(() => {});
      $('viewport').classList.add('is-live');
      $('camHint').textContent = '';
      requestAnimationFrame(() => { sizeOverlay(); setLive(true); });
    } catch (e) {
      toast('Camera permission denied', true);
    }
  }

  function stopCamera() {
    if (state.live.stream) {
      state.live.stream.getTracks().forEach((t) => t.stop());
      state.live.stream = null;
    }
    $('video').srcObject = null;
    $('viewport').classList.remove('is-live');
    $('camHint').textContent = 'Camera permission is needed to scan walls.';
  }

  function setLive(on) {
    state.live.on = !!on;
    const pill = $('livePill');
    $('liveLabel').textContent = on ? 'Live on' : 'Live off';
    $('btnLive').style.color = on ? 'var(--accent)' : '';
    if (!on) {
      pill.textContent = 'STANDBY';
      pill.className = 'hud-pill';
      $('liveReadout').textContent = 'Start the camera to scan a wall';
      $('wallPill').hidden = true;
      clearOverlay();
      return;
    }
    pill.textContent = 'LIVE';
    pill.className = 'hud-pill is-live';
    tickLive();
  }

  function clearOverlay() {
    const c = $('overlay');
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  }

  // Grab the current video frame. `maxSide` keeps the payload small.
  function grabFrame(maxSide, quality) {
    const video = $('video');
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    const scale = Math.min(1, (maxSide || 960) / Math.max(vw, vh));
    const w = Math.max(2, Math.round(vw * scale));
    const h = Math.max(2, Math.round(vh * scale));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    return {
      w: w,
      h: h,
      canvas: canvas,
      dataUrl: canvas.toDataURL('image/jpeg', quality || 0.72)
    };
  }

  function sizeOverlay() {
    const c = $('overlay');
    const vp = $('viewport');
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    c.width = Math.max(1, Math.round(vp.clientWidth * dpr));
    c.height = Math.max(1, Math.round(vp.clientHeight * dpr));
  }

  // Draw server contours on top of the preview. Frame coords → cover-fit screen.
  function drawContours(contours, level, frameW, frameH) {
    const c = $('overlay');
    if (!c.width) sizeOverlay();
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!contours || !contours.length || !frameW || !frameH) return;

    const scale = Math.max(c.width / frameW, c.height / frameH);
    const offX = (c.width - frameW * scale) / 2;
    const offY = (c.height - frameH * scale) / 2;
    const color = LEVEL_COLOR[level] || '#ff3b30';

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    contours.forEach((ct) => {
      const pts = ct.points || [];
      if (pts.length < 2) return;
      const path = new Path2D();
      pts.forEach((p, i) => {
        const x = p[0] * scale + offX;
        const y = p[1] * scale + offY;
        if (i === 0) path.moveTo(x, y);
        else path.lineTo(x, y);
      });
      // dark halo for legibility over any wall colour, then the severity colour
      ctx.strokeStyle = 'rgba(4, 12, 22, 0.72)';
      ctx.lineWidth = 7;
      ctx.stroke(path);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3.2;
      ctx.stroke(path);
    });
  }

  function setHud(level, statusText, masonry) {
    const pill = $('livePill');
    pill.textContent = (LEVEL_NAME[level] || 'LIVE');
    pill.className = 'hud-pill is-' + (['safe', 'minor', 'medium', 'severe'][level] || '');
    const wall = $('wallPill');
    wall.hidden = !masonry;
    $('liveReadout').textContent = statusText;
  }

  // ── live analysis loop ───────────────────────────────────────────
  function tickLive() {
    if (!state.live.on) return;
    clearTimeout(state.live.timer);
    state.live.timer = setTimeout(async () => {
      if (!state.live.on) return;
      if (!state.connected || state.live.busy || document.hidden) return tickLive();
      if (!state.live.stream) return tickLive();

      state.live.busy = true;
      try {
        const f = grabFrame(900, 0.7);
        state.frame = { w: f.w, h: f.h };
        const res = await api('/analyze_live', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: f.dataUrl })
        }, 15000);
        state.live.last = res;
        drawContours(res.contours, res.level, f.w, f.h);
        const len = Math.round(res.crack_length || 0);
        const w = (res.crack_width || 0).toFixed(1);
        setHud(res.level, (res.status || '') + ' · ' + len + ' px · ' + w + ' px', res.masonry);
      } catch (e) {
        if (e.offline) {
          setConn(false, '');
          setLive(false);
          toast('Lost the server — live scan stopped', true);
        }
      } finally {
        state.live.busy = false;
        tickLive();
      }
    }, 950);
  }

  function stopLive(quiet) {
    state.live.on = false;
    clearTimeout(state.live.timer);
    if (!quiet) {
      $('liveLabel').textContent = 'Live off';
      $('livePill').textContent = 'STANDBY';
      $('livePill').className = 'hud-pill';
      $('wallPill').hidden = true;
      $('liveReadout').textContent = 'Start the camera to scan a wall';
      clearOverlay();
    }
  }

  // ── scanning a photo ─────────────────────────────────────────────
  async function scanBlob(blob, name) {
    if (!state.server) {
      showScreen('settings');
      toast('Set your server address first', true);
      return;
    }
    const btn = $('btnShutter');
    btn.classList.add('is-busy');
    try {
      const form = new FormData();
      form.append('image', blob, name || 'scan.jpg');
      const rec = await api('/api/mobile/scan', { method: 'POST', body: form }, 120000);
      if (!rec || rec.ok === false) throw new Error((rec && rec.error) || 'Scan failed');
      showResult(rec);
    } catch (e) {
      toast(e.message || 'Scan failed', true);
    } finally {
      btn.classList.remove('is-busy');
    }
  }

  async function scanFrame() {
    if (!state.live.stream) {
      toast('Enable the camera first');
      return;
    }
    if (!$('video').videoWidth) {
      toast('Camera is still starting…');
      return;
    }
    const f = grabFrame(1600, 0.9);
    const blob = await new Promise((resolve) => f.canvas.toBlob(resolve, 'image/jpeg', 0.9));
    if (!blob) {
      toast('Could not grab a frame', true);
      return;
    }
    await scanBlob(blob, 'scan_' + Date.now() + '.jpg');
  }

  async function scanFromGallery() {
    const Cam = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Camera;
    if (Cam && Cam.getPhoto) {
      try {
        const photo = await Cam.getPhoto({
          quality: 88,
          resultType: 'base64',
          source: 'PHOTOS',
          width: 2400,
          correctOrientation: true
        });
        const uri = 'data:image/' + (photo.format || 'jpeg') + ';base64,' + photo.base64String;
        await scanDataUri(uri);
        return;
      } catch (e) {
        if (e && /cancel/i.test(e.message || '')) return;
        // fall through to the file picker
      }
    }
    openFilePicker();
  }

  async function takePhoto() {
    const Cam = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Camera;
    if (Cam && Cam.getPhoto) {
      try {
        const photo = await Cam.getPhoto({
          quality: 88,
          resultType: 'base64',
          source: 'CAMERA',
          width: 2400,
          correctOrientation: true
        });
        const uri = 'data:image/' + (photo.format || 'jpeg') + ';base64,' + photo.base64String;
        await scanDataUri(uri);
        return;
      } catch (e) {
        if (e && /cancel/i.test(e.message || '')) return;
      }
    }
    openFilePicker();
  }

  function openFilePicker() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      await scanBlob(file, file.name || 'scan.jpg');
    };
    input.click();
  }

  async function scanDataUri(uri) {
    try {
      const res = await api('/api/mobile/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: uri })
      }, 120000);
      if (!res || res.ok === false) throw new Error((res && res.error) || 'Scan failed');
      showResult(res);
    } catch (e) {
      toast(e.message || 'Scan failed', true);
    }
  }

  // ── result sheet ─────────────────────────────────────────────────
  function sevClass(level) {
    return level >= 3 ? 'l3' : level === 2 ? 'l2' : level === 1 ? 'l1' : '';
  }

  function showResult(rec) {
    state.current = rec;
    const level = typeof rec.level === 'number' ? rec.level : levelFromSeverity(rec.severity);
    const img = rec.annotated_image || rec.image_url || '';
    const sheetImg = $('sheetImg');
    sheetImg.src = img;
    sheetImg.alt = 'Annotated crack scan';

    const sev = $('sheetSev');
    sev.textContent = (LEVEL_NAME[level] || 'RESULT');
    sev.className = 'sev-badge ' + sevClass(level);

    $('sheetTitle').textContent = 'Scan result';
    $('sheetSub').textContent = rec.timestamp || 'Just now';
    $('mLength').textContent = fmtCm(rec.length_cm);
    $('mWidth').textContent = fmtCm(rec.width_cm);
    $('mCount').textContent = String(rec.cracks != null ? rec.cracks : '—');

    const wall = $('sheetWall');
    wall.hidden = !rec.masonry;
    wall.textContent = 'BRICK WALL';

    const insp = rec.inspection_status || 'new';
    $$('#inspectRow .chip').forEach((c) => c.classList.toggle('is-on', c.dataset.insp === insp));
    $('inspectRow').hidden = !rec.id;
    $('btnDeleteScan').hidden = !rec.id;

    $('sheet').hidden = false;
    document.body.style.overflow = 'hidden';
  }

  function levelFromSeverity(sev) {
    return { safe: 0, minor: 1, moderate: 2, severe: 3 }[sev] || 0;
  }

  function fmtCm(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (v < 10 ? v.toFixed(1) : Math.round(v)) + ' cm';
  }

  function closeSheet() {
    $('sheet').hidden = true;
    document.body.style.overflow = '';
  }

  async function patchInspection(status) {
    if (!state.current || !state.current.id) return;
    try {
      await api('/api/cracks/' + state.current.id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inspection_status: status })
      }, 15000);
      state.current.inspection_status = status;
      $$('#inspectRow .chip').forEach((c) => c.classList.toggle('is-on', c.dataset.insp === status));
      toast('Marked ' + status.replace('_', ' '));
      if ($('screen-history').classList.contains('is-active')) loadHistory();
    } catch (e) {
      toast(e.message || 'Update failed', true);
    }
  }

  async function deleteCurrent() {
    if (!state.current || !state.current.id) return;
    if (!window.confirm('Delete this scan from the history?')) return;
    try {
      await api('/api/mobile/records/' + state.current.id, { method: 'DELETE' }, 15000);
      closeSheet();
      toast('Scan deleted');
      loadHistory();
    } catch (e) {
      toast(e.message || 'Delete failed', true);
    }
  }

  // ── history ──────────────────────────────────────────────────────
  async function loadHistory() {
    const list = $('histList');
    const empty = $('histEmpty');
    if (!state.server) {
      list.innerHTML = '';
      empty.hidden = false;
      $('histMeta').textContent = 'Connect a server to see saved scans';
      return;
    }
    try {
      let q = '/api/cracks?limit=60&images=0';
      if (state.filter) q += '&inspection_status=' + encodeURIComponent(state.filter);
      const data = await api(q, { method: 'GET' }, 15000);
      const rows = (data && data.cracks) || [];
      list.innerHTML = '';
      $('histMeta').textContent = rows.length
        ? rows.length + (rows.length === 1 ? ' scan' : ' scans') + ' on this server'
        : 'Saved scans from this server';
      empty.hidden = rows.length > 0;
      rows.forEach((r) => list.appendChild(recordRow(r)));
    } catch (e) {
      list.innerHTML = '';
      empty.hidden = false;
      $('histMeta').textContent = 'Could not load history';
      if (e.offline) setConn(false, '');
    }
  }

  function recordRow(r) {
    const level = levelFromSeverity(r.severity);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'rec';

    const img = document.createElement('img');
    img.className = 'rec-thumb';
    img.loading = 'lazy';
    img.alt = '';
    if (r.image_url) img.src = r.image_url;

    const body = document.createElement('div');
    body.className = 'rec-body';
    const title = document.createElement('div');
    title.className = 'rec-title';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = LEVEL_COLOR[level];
    const name = document.createElement('span');
    name.textContent = SEV_NAME[r.severity] || 'CRACK';
    title.appendChild(dot);
    title.appendChild(name);

    const sub = document.createElement('div');
    sub.className = 'rec-sub';
    sub.textContent = [
      (r.length_cm != null ? r.length_cm + ' cm long' : ''),
      (r.width_cm != null ? r.width_cm + ' cm wide' : ''),
      shortDate(r.timestamp)
    ].filter(Boolean).join(' · ');

    const state2 = document.createElement('div');
    state2.className = 'rec-sub';
    state2.textContent = 'Inspection: ' + (r.inspection_status || 'new').replace('_', ' ');

    body.appendChild(title);
    body.appendChild(sub);
    body.appendChild(state2);

    const chev = document.createElement('span');
    chev.className = 'rec-chev';
    chev.textContent = '›';

    btn.appendChild(img);
    btn.appendChild(body);
    btn.appendChild(chev);
    btn.addEventListener('click', () => {
      showResult({
        id: r.id,
        level: level,
        severity: r.severity,
        length_cm: r.length_cm,
        width_cm: r.width_cm,
        cracks: null,
        masonry: false,
        image_url: r.image_url,
        timestamp: shortDate(r.timestamp),
        inspection_status: r.inspection_status
      });
    });
    return btn;
  }

  function shortDate(ts) {
    if (!ts) return '';
    // "2026-11-07 16:13:55" → "7 Nov, 16:13"
    const m = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/.exec(ts);
    if (!m) return ts;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return parseInt(m[3], 10) + ' ' + months[parseInt(m[2], 10) - 1] + ', ' + m[4] + ':' + m[5];
  }

  // ── settings ─────────────────────────────────────────────────────
  function loadSettings() {
    $('serverUrl').value = state.server || '';
    $('kvPlatform').textContent = (isNative ? 'Native (' + state.platform + ')' : 'Web preview');
    $('serverStatus').textContent = state.server
      ? (state.connected ? 'Connected to ' + (state.health && state.health.host || state.server) : 'Saved, not reachable')
      : 'Not tested yet';
    $('serverStatus').className = 'status-line ' + (state.connected ? 'is-ok' : '');
  }

  async function saveServer(ev) {
    if (ev) ev.preventDefault();
    const url = normalizeServer($('serverUrl').value);
    if (!url) {
      toast('Enter the server address', true);
      return;
    }
    state.server = url;
    localStorage.setItem(LS_SERVER, url);
    $('serverStatus').textContent = 'Testing ' + url + ' …';
    $('serverStatus').className = 'status-line';
    const ok = await connect(true);
    $('serverStatus').textContent = ok
      ? 'Connected · ' + (state.health.app || 'CrackScan') + ' v' + state.health.version
      : 'No answer from ' + url;
    $('serverStatus').className = 'status-line ' + (ok ? 'is-ok' : 'is-bad');
    if (ok) {
      toast('Server connected');
      loadHistory();
    } else {
      toast('Could not reach that server', true);
    }
  }

  async function detectAgain() {
    const guess = defaultServer();
    if (guess) {
      $('serverUrl').value = guess;
      await saveServer();
      return;
    }
    toast('Enter your PC\'s LAN address, e.g. 192.168.1.20:5000');
    $('serverUrl').focus();
  }

  // ── wiring ───────────────────────────────────────────────────────
  function init() {
    state.platform = (window.Capacitor && window.Capacitor.getPlatform && window.Capacitor.getPlatform()) || 'web';
    state.server = normalizeServer(localStorage.getItem(LS_SERVER) || defaultServer());

    $$('#tabbar .tab').forEach((tab) => {
      tab.addEventListener('click', () => showScreen(tab.dataset.screen));
    });

    $('btnReconnect').addEventListener('click', () => {
      if (state.connected) showScreen('settings');
      else {
        showScreen('settings');
        saveServer();
      }
    });

    $('btnEnableCam').addEventListener('click', enableCamera);
    $('btnGallery').addEventListener('click', scanFromGallery);
    $('btnLive').addEventListener('click', () => {
      if (!state.live.stream) { enableCamera(); return; }
      setLive(!state.live.on);
    });

    $('serverForm').addEventListener('submit', saveServer);
    $('btnRescan').addEventListener('click', detectAgain);
    $('btnRefreshHist').addEventListener('click', loadHistory);

    $$('#histFilters .chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        state.filter = chip.dataset.filter;
        $$('#histFilters .chip').forEach((c) => c.classList.toggle('is-on', c === chip));
        loadHistory();
      });
    });

    $$('[data-close-sheet]').forEach((el) => el.addEventListener('click', closeSheet));
    $('btnNewScan').addEventListener('click', () => {
      closeSheet();
      showScreen('scan');
      if (!state.live.stream) enableCamera();
    });
    $('btnDeleteScan').addEventListener('click', deleteCurrent);
    $$('#inspectRow .chip').forEach((chip) => {
      chip.addEventListener('click', () => patchInspection(chip.dataset.insp));
    });

    // Long-press shutter → native camera (full-resolution photo). A long press
    // must not also fire the normal tap handler when the finger lifts.
    let pressTimer = null;
    let suppressClick = false;
    const shutter = $('btnShutter');
    shutter.addEventListener('click', () => {
      if (suppressClick) { suppressClick = false; return; }
      scanFrame();
    });
    shutter.addEventListener('pointerdown', () => {
      suppressClick = false;
      pressTimer = setTimeout(() => { pressTimer = null; suppressClick = true; takePhoto(); }, 650);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) =>
      shutter.addEventListener(ev, () => { if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } })
    );

    window.addEventListener('resize', () => { if ($('viewport').classList.contains('is-live')) sizeOverlay(); });

    // Android hardware back: close the sheet first, then fall back to the OS.
    const AppPlugin = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.App;
    if (AppPlugin && AppPlugin.addListener) {
      AppPlugin.addListener('backButton', () => {
        if (!$('sheet').hidden) closeSheet();
        else if ($('screen-scan').classList.contains('is-active') && state.live.on) stopLive();
        else AppPlugin.exitApp();
      });
    }

    document.addEventListener('visibilitychange', () => {
      if (document.hidden && state.live.on) stopLive(true);
    });

    loadSettings();
    connect(true).then((ok) => { if (ok) loadHistory(); });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
