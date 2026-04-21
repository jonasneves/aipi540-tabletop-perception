// WebRTC video-track pairing via signal.neevs.io.
// Desktop waits for phone. Phone sends camera video over a one-way stream.
// Signaling protocol matches catwatcher (proven in prod):
//   {type:'signal', peer:<role>, data:{offer|answer|ice}}  for per-peer messages
//   {type:'state',  peers:{<role>: lastSignal}}            for late-joiner snapshots
//   {type:'ping'}                                          heartbeat

const SIGNAL_URL = 'wss://signal.neevs.io';
const ICE = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun.cloudflare.com:3478' },
];
const CONN_TIMEOUT_MS = 20000;

function log(...a) { try { console.log('[webrtc]', ...a); } catch {} }

function openSignaling(roomId, role) {
  const ws = new WebSocket(`${SIGNAL_URL}/tp-${roomId}/ws`);
  const heartbeat = setInterval(() => {
    if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
  }, 20000);
  ws._role = role;
  ws._send = (data) => {
    if (ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'signal', peer: role, data }));
      log(role, 'sent', Object.keys(data).join(','));
    } else {
      log(role, 'send dropped, ws state=', ws.readyState);
    }
  };
  ws._close = () => {
    clearInterval(heartbeat);
    try { ws.close(); } catch {}
  };
  return ws;
}

// ── Desktop: waits for phone to dial in ───────────────────────
export function startDesktop(roomId, callbacks = {}) {
  const { onRemoteStream, onStatus, onConnected } = callbacks;
  const ws = openSignaling(roomId, 'desktop');
  let pc = null;
  const pendingIce = [];

  const ensurePc = () => {
    if (pc) return pc;
    pc = new RTCPeerConnection({ iceServers: ICE });
    pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
    pc.ontrack = (e) => { log('desktop', 'ontrack'); onRemoteStream && onRemoteStream(e.streams[0]); };
    pc.oniceconnectionstatechange = () => {
      const st = pc.iceConnectionState;
      log('desktop', 'iceState=', st);
      onStatus && onStatus(st);
      if (st === 'connected' || st === 'completed') onConnected && onConnected();
      if (st === 'failed') { log('desktop', 'restart ICE'); pc.restartIce(); }
    };
    return pc;
  };

  const handle = async (data) => {
    if (!data) return;
    if (data.offer) {
      ensurePc();
      await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
      for (const c of pendingIce) { try { await pc.addIceCandidate(c); } catch {} }
      pendingIce.length = 0;
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      ws._send({ answer });
    } else if (data.ice) {
      const cand = new RTCIceCandidate(data.ice);
      if (pc && pc.remoteDescription) { try { await pc.addIceCandidate(cand); } catch (e) { log('ice add err', e); } }
      else pendingIce.push(cand);
    }
  };

  ws.onopen = () => { log('desktop', 'ws open, role=desktop, room=', roomId); onStatus && onStatus('waiting for phone'); };
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'signal' && msg.peer !== 'desktop') handle(msg.data).catch((e) => log('handle err', e));
    if (msg.type === 'state') {
      const peers = msg.peers || {};
      for (const k of Object.keys(peers)) if (k !== 'desktop') handle(peers[k]).catch((e) => log('handle err', e));
    }
  };
  ws.onerror = (e) => { log('desktop', 'ws err', e); onStatus && onStatus('signaling error'); };
  ws.onclose = () => { log('desktop', 'ws close'); onStatus && onStatus('signaling closed'); };

  return { stop: () => { ws._close(); if (pc) { pc.close(); pc = null; } } };
}

// ── Phone: dials desktop, sends video ────────────────────────
export async function startPhone(roomId, stream, callbacks = {}) {
  const { onStatus } = callbacks;
  const ws = openSignaling(roomId, 'phone');
  const pc = new RTCPeerConnection({ iceServers: ICE });
  const pendingIce = [];
  let negotiating = false;
  let connectionTimer = null;

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));
  pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
  pc.oniceconnectionstatechange = () => {
    const st = pc.iceConnectionState;
    log('phone', 'iceState=', st);
    onStatus && onStatus(st);
    if (st === 'connected' || st === 'completed') {
      if (connectionTimer) { clearTimeout(connectionTimer); connectionTimer = null; }
    }
    if (st === 'failed') {
      log('phone', 'ICE failed, restarting');
      pc.restartIce();
      renegotiate();
    }
  };

  const renegotiate = async () => {
    if (negotiating) return;
    negotiating = true;
    try {
      const offer = await pc.createOffer({ iceRestart: true });
      await pc.setLocalDescription(offer);
      ws._send({ offer });
    } finally { negotiating = false; }
  };

  const handle = async (data) => {
    if (!data) return;
    if (data.answer) {
      log('phone', 'got answer');
      await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
      for (const c of pendingIce) { try { await pc.addIceCandidate(c); } catch {} }
      pendingIce.length = 0;
    } else if (data.ice) {
      const cand = new RTCIceCandidate(data.ice);
      if (pc.remoteDescription) { try { await pc.addIceCandidate(cand); } catch (e) { log('ice add err', e); } }
      else pendingIce.push(cand);
    }
  };

  // Set handlers BEFORE anything else so incoming messages don't race the offer.
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'signal' && msg.peer !== 'phone') handle(msg.data).catch((e) => log('handle err', e));
    if (msg.type === 'state') {
      const peers = msg.peers || {};
      for (const k of Object.keys(peers)) if (k !== 'phone') handle(peers[k]).catch((e) => log('handle err', e));
    }
  };
  ws.onerror = (e) => { log('phone', 'ws err', e); onStatus && onStatus('signaling error'); };
  ws.onclose = () => { log('phone', 'ws close'); onStatus && onStatus('signaling closed'); };

  await new Promise((res, rej) => {
    ws.onopen = () => { log('phone', 'ws open'); res(); };
    setTimeout(() => rej(new Error('signaling timeout')), 10000);
  });

  // Create and send initial offer.
  negotiating = true;
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws._send({ offer });
  negotiating = false;
  onStatus && onStatus('offer sent, waiting for answer');

  connectionTimer = setTimeout(() => {
    if (pc.iceConnectionState === 'new' || pc.iceConnectionState === 'checking') {
      log('phone', 'connection timeout'); onStatus && onStatus('connection timeout — check network');
    }
  }, CONN_TIMEOUT_MS);

  return { stop: () => { ws._close(); pc.close(); stream.getTracks().forEach((t) => t.stop()); if (connectionTimer) clearTimeout(connectionTimer); } };
}

export function makeRoomId() {
  const a = 'abcdefghjkmnpqrstuvwxyz23456789';
  let s = '';
  for (let i = 0; i < 6; i++) s += a[Math.floor(Math.random() * a.length)];
  return s;
}
