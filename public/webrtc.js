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
const CONN_TIMEOUT_MS = 30000;          // give pairing a real chance before failing the user out
const DISCONNECT_GRACE_MS = 10000;      // most ICE 'disconnected' events recover on their own

function log(...a) { try { console.log('[webrtc]', ...a); } catch {} }

function openSignaling(roomId, role) {
  const ws = new WebSocket(`${SIGNAL_URL}/tp-${roomId}/ws`);
  const heartbeat = setInterval(() => {
    if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
  }, 20000);
  ws._role = role;
  ws._roomId = roomId;
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

// Visibility recovery: iOS Safari kills idle WebSockets when the tab
// backgrounds. The 20s heartbeat can't save it. On visibilitychange→visible,
// if our WS is dead, install a fresh one against the same room and rewire
// the message handler. The PC itself survives — we just need a working
// signaling lane to send the ICE-restart offer through.
function installVisibilityRecovery({ getWs, setWs, roomId, role, onMessage, onError, onReconnect }) {
  if (typeof document === 'undefined') return () => {};
  let reopening = false;
  const handler = () => {
    if (document.visibilityState !== 'visible') return;
    const ws = getWs();
    if (!ws || ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) return;
    if (reopening) return;
    reopening = true;
    log(role, 'visibility: ws dead, reopening');
    const fresh = openSignaling(roomId, role);
    fresh.onopen = () => {
      log(role, 'visibility: fresh ws open');
      const old = getWs();
      setWs(fresh);
      reopening = false;
      try { old && old._close && old._close(); } catch {}
      try { onReconnect && onReconnect(fresh); } catch {}
    };
    fresh.onmessage = onMessage;
    fresh.onerror   = (e) => { reopening = false; onError && onError(e); };
  };
  document.addEventListener('visibilitychange', handler);
  return () => document.removeEventListener('visibilitychange', handler);
}

// ── Desktop: waits for phone to dial in ───────────────────────
export function startDesktop(roomId, callbacks = {}) {
  const { onRemoteStream, onStatus, onConnected } = callbacks;
  let ws = openSignaling(roomId, 'desktop');
  let pc = null;
  let graceTimer = null;
  const pendingIce = [];

  const ensurePc = () => {
    if (pc) return pc;
    pc = new RTCPeerConnection({ iceServers: ICE });
    pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
    pc.ontrack = (e) => { log('desktop', 'ontrack'); onRemoteStream && onRemoteStream(e.streams[0]); };
    pc.oniceconnectionstatechange = () => {
      const st = pc.iceConnectionState;
      log('desktop', 'iceState=', st);
      if (st === 'connected' || st === 'completed') {
        if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
        onStatus && onStatus('connected');
        onConnected && onConnected();
      } else if (st === 'disconnected') {
        // Most disconnects (tab unfocused, brief wifi glitch) recover on their own.
        // Wait a bit before bothering the user — and before doing anything that
        // could make things worse.
        onStatus && onStatus('reconnecting — waiting for path to recover…');
        if (!graceTimer) {
          graceTimer = setTimeout(() => {
            graceTimer = null;
            if (pc && pc.iceConnectionState === 'disconnected') {
              log('desktop', 'still disconnected after grace, awaiting phone restart');
              onStatus && onStatus('reconnecting — waiting on phone to re-offer…');
            }
          }, DISCONNECT_GRACE_MS);
        }
      } else if (st === 'failed') {
        log('desktop', 'ICE failed, asking phone to restart');
        onStatus && onStatus('reconnecting — restarting connection…');
      }
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

  const wireWs = (sock) => {
    sock.onopen = () => { log('desktop', 'ws open, room=', roomId); onStatus && onStatus('waiting for phone'); };
    sock.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'signal' && msg.peer !== 'desktop') handle(msg.data).catch((e) => log('handle err', e));
      if (msg.type === 'state') {
        const peers = msg.peers || {};
        for (const k of Object.keys(peers)) if (k !== 'desktop') handle(peers[k]).catch((e) => log('handle err', e));
      }
    };
    sock.onerror = (e) => { log('desktop', 'ws err', e); onStatus && onStatus('signaling error'); };
    sock.onclose = () => log('desktop', 'ws close');
  };
  wireWs(ws);

  // iOS Safari kills idle WebSockets when phone tab backgrounds. When the
  // user re-foregrounds, recovery should be ~1s, not 20s of ICE timeout.
  const detachVis = installVisibilityRecovery({
    getWs: () => ws,
    setWs: (next) => { ws = next; wireWs(ws); },
    roomId, role: 'desktop',
    onMessage: (ev) => ws.onmessage(ev),
    onError:   (e)  => log('desktop', 'visibility-reopen err', e),
    onReconnect: () => onStatus && onStatus('signaling restored'),
  });

  return { stop: () => {
    detachVis();
    if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
    ws._close();
    if (pc) { pc.close(); pc = null; }
  } };
}

// ── Phone: dials desktop, sends video ────────────────────────
export async function startPhone(roomId, stream, callbacks = {}) {
  const { onStatus } = callbacks;
  let ws = openSignaling(roomId, 'phone');
  const pc = new RTCPeerConnection({ iceServers: ICE });
  const pendingIce = [];
  let negotiating = false;
  let connectionTimer = null;
  let graceTimer = null;

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));
  pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
  pc.oniceconnectionstatechange = () => {
    const st = pc.iceConnectionState;
    log('phone', 'iceState=', st);
    if (st === 'checking')  { onStatus && onStatus('finding network path…'); }
    if (st === 'connected' || st === 'completed') {
      if (connectionTimer) { clearTimeout(connectionTimer); connectionTimer = null; }
      if (graceTimer)      { clearTimeout(graceTimer);      graceTimer = null; }
      onStatus && onStatus('connected');
    }
    if (st === 'disconnected') {
      // Wait out transient drops before doing anything.
      onStatus && onStatus('reconnecting — waiting on path…');
      if (!graceTimer) {
        graceTimer = setTimeout(() => {
          graceTimer = null;
          if (pc.iceConnectionState === 'disconnected') {
            log('phone', 'still disconnected after grace, restarting');
            try { pc.restartIce(); } catch {}
            renegotiate();
          }
        }, DISCONNECT_GRACE_MS);
      }
    }
    if (st === 'failed') {
      log('phone', 'ICE failed, restarting');
      onStatus && onStatus('reconnecting — restarting connection…');
      try { pc.restartIce(); } catch {}
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

  const wireWs = (sock) => {
    // Set message handler BEFORE anything else so incoming messages don't race the offer.
    sock.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'signal' && msg.peer !== 'phone') handle(msg.data).catch((e) => log('handle err', e));
      if (msg.type === 'state') {
        const peers = msg.peers || {};
        for (const k of Object.keys(peers)) if (k !== 'phone') handle(peers[k]).catch((e) => log('handle err', e));
      }
    };
    sock.onerror = (e) => { log('phone', 'ws err', e); onStatus && onStatus('signaling error'); };
    sock.onclose = () => log('phone', 'ws close');
  };
  wireWs(ws);

  await new Promise((res, rej) => {
    ws.onopen = () => { log('phone', 'ws open'); onStatus && onStatus('signal channel open'); res(); };
    setTimeout(() => rej(new Error('signaling timeout — close and reopen the desktop QR')), 10000);
  });

  // Create and send initial offer.
  negotiating = true;
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws._send({ offer });
  negotiating = false;
  onStatus && onStatus('offer sent, waiting for desktop…');

  connectionTimer = setTimeout(() => {
    if (pc.iceConnectionState === 'new' || pc.iceConnectionState === 'checking') {
      log('phone', 'connection timeout');
      onStatus && onStatus('connection timeout — close and rescan the desktop QR');
    }
  }, CONN_TIMEOUT_MS);

  // Visibility recovery: when the phone tab is backgrounded, iOS kills the
  // signal WS. Reopen on return, rewire handlers, then trigger ICE restart.
  const detachVis = installVisibilityRecovery({
    getWs: () => ws,
    setWs: (next) => { ws = next; wireWs(ws); },
    roomId, role: 'phone',
    onMessage: (ev) => ws.onmessage(ev),
    onError:   (e)  => log('phone', 'visibility-reopen err', e),
    onReconnect: () => { onStatus && onStatus('signaling restored, restarting connection…'); renegotiate(); },
  });

  return { stop: () => {
    detachVis();
    if (graceTimer)      { clearTimeout(graceTimer); graceTimer = null; }
    if (connectionTimer) { clearTimeout(connectionTimer); connectionTimer = null; }
    ws._close();
    pc.close();
    stream.getTracks().forEach((t) => t.stop());
  } };
}

export function makeRoomId() {
  const a = 'abcdefghjkmnpqrstuvwxyz23456789';
  let s = '';
  for (let i = 0; i < 6; i++) s += a[Math.floor(Math.random() * a.length)];
  return s;
}
