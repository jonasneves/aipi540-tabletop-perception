// WebRTC video-track pairing via signal.neevs.io.
// Desktop waits for phone. Phone sends camera video over a one-way stream.
// Signaling protocol:
//   {type:'signal', peer:<peerId>, data:{offer|answer|ice}}  per-peer messages
//   {type:'state',  peers:{<peerId>: lastSignal}}            late-joiner snapshots
//   {type:'ping'}                                            heartbeat
//
// Reliability notes:
// - peerId = role + '-' + nonce, so rescanning the QR or opening a duplicate
//   tab doesn't collide under a fixed role key.
// - Desktop closes+recreates the RTCPeerConnection on every incoming offer.
//   Prevents InvalidStateError when the phone restarts ICE on top of a pc
//   that's already in have-local-answer.
// - State snapshots only replay offer/answer, never stale ICE candidates.
//   They're also skipped entirely once the connection is healthy, so a WS
//   re-open (visibility recovery) doesn't blow up a working pc.
// - Phone's initial offer and ICE-restart offer share one renegotiate()
//   path to prevent the negotiating flag from racing.
// - ?debug in the URL enables a log sink for the in-page debug panel.

const SIGNAL_URL = 'wss://signal.neevs.io';
// STUN lets peers discover their public address. TURN relays media when
// direct paths fail (symmetric NAT on cellular, corporate wifi, guest wifi).
// OpenRelay is Metered.ca's public demo TURN — shared credentials, rate-
// limited, best-effort uptime. Fine for classroom demos; replace with a
// private TURN key if you hit quota or need reliable capacity.
const ICE = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun.cloudflare.com:3478' },
  {
    urls: [
      'turn:openrelay.metered.ca:80',
      'turn:openrelay.metered.ca:443',
      'turn:openrelay.metered.ca:443?transport=tcp',
    ],
    username: 'openrelayproject',
    credential: 'openrelayproject',
  },
];
const CONN_TIMEOUT_MS = 30000;
const DISCONNECT_GRACE_MS = 10000;

export const DEBUG = typeof location !== 'undefined' && /[?&]debug(=|&|$)/.test(location.search);
const logSinks = new Set();
export function onDebugLog(fn) { logSinks.add(fn); return () => logSinks.delete(fn); }

function log(...a) {
  try { console.log('[webrtc]', ...a); } catch {}
  if (!logSinks.size) return;
  const ts = new Date().toISOString().slice(11, 23);
  const msg = a.map(x => {
    if (typeof x === 'string') return x;
    try { return JSON.stringify(x); } catch { return String(x); }
  }).join(' ');
  for (const fn of logSinks) { try { fn(`${ts} ${msg}`); } catch {} }
}

function makePeerId(role) {
  return role + '-' + Math.random().toString(36).slice(2, 8);
}

function openSignaling(roomId, peerId) {
  const ws = new WebSocket(`${SIGNAL_URL}/tp-${roomId}/ws`);
  const heartbeat = setInterval(() => {
    if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
  }, 20000);
  ws._peerId = peerId;
  ws._roomId = roomId;
  ws._send = (data) => {
    if (ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'signal', peer: peerId, data }));
      log(peerId, 'sent', Object.keys(data).join(','));
    } else {
      log(peerId, 'send dropped, ws state=', ws.readyState);
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
function installVisibilityRecovery({ getWs, setWs, roomId, peerId, onMessage, onError, onReconnect }) {
  if (typeof document === 'undefined') return () => {};
  let reopening = false;
  const handler = () => {
    if (document.visibilityState !== 'visible') return;
    const ws = getWs();
    if (!ws || ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) return;
    if (reopening) return;
    reopening = true;
    log(peerId, 'visibility: ws dead, reopening');
    const fresh = openSignaling(roomId, peerId);
    fresh.addEventListener('open', () => {
      log(peerId, 'visibility: fresh ws open');
      const old = getWs();
      setWs(fresh);
      reopening = false;
      try { old && old._close && old._close(); } catch {}
      try { onReconnect && onReconnect(fresh); } catch {}
    }, { once: true });
    fresh.onmessage = onMessage;
    fresh.onerror   = (e) => { reopening = false; onError && onError(e); };
  };
  document.addEventListener('visibilitychange', handler);
  return () => document.removeEventListener('visibilitychange', handler);
}

// State snapshots may contain offers/answers from prior sessions or dead
// peers. Apply only the semantic-describe messages (offer/answer) and drop
// stale ICE — those will be rejected by a fresh pc anyway.
function extractFromState(peers, selfPeerId, otherRolePrefix) {
  const out = [];
  for (const k of Object.keys(peers || {})) {
    if (k === selfPeerId) continue;
    if (otherRolePrefix && !k.startsWith(otherRolePrefix + '-')) continue;
    const d = peers[k];
    if (!d) continue;
    if (d.offer || d.answer) out.push(d);
  }
  return out;
}

function summariseStats(pc) {
  return pc.getStats().then(stats => {
    const pair = [...stats.values()].find(s => s.type === 'candidate-pair' && s.state === 'succeeded' && s.nominated);
    if (!pair) return null;
    const local = stats.get(pair.localCandidateId);
    const remote = stats.get(pair.remoteCandidateId);
    const candInfo = (c) => c ? {
      type: c.candidateType,
      protocol: c.protocol,
      address: c.address || c.ip,
      port: c.port,
      relayProtocol: c.relayProtocol,
    } : null;
    return {
      bytesReceived: pair.bytesReceived || 0,
      bytesSent: pair.bytesSent || 0,
      currentRoundTripTime: pair.currentRoundTripTime || 0,
      local: candInfo(local),
      remote: candInfo(remote),
    };
  }).catch(() => null);
}

// ── Desktop: waits for phone to dial in ───────────────────────
export function startDesktop(roomId, callbacks = {}) {
  const { onRemoteStream, onStatus, onConnected } = callbacks;
  const peerId = makePeerId('desktop');
  let ws = openSignaling(roomId, peerId);
  let pc = null;
  let graceTimer = null;
  let pendingIce = [];

  const resetPc = (reason) => {
    if (pc) {
      log('desktop', 'closing pc:', reason, 'state=', pc.signalingState);
      try { pc.close(); } catch {}
      pc = null;
    }
    pendingIce = [];
  };

  const isConnected = () => pc && (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed');

  const ensurePc = () => {
    if (pc) return pc;
    pc = new RTCPeerConnection({ iceServers: ICE });
    pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
    pc.ontrack = (e) => { log('desktop', 'ontrack'); onRemoteStream && onRemoteStream(e.streams[0]); };
    pc.onsignalingstatechange  = () => log('desktop', 'signalingState=', pc.signalingState);
    pc.onconnectionstatechange = () => log('desktop', 'connectionState=', pc.connectionState);
    pc.oniceconnectionstatechange = () => {
      if (!pc) return;
      const st = pc.iceConnectionState;
      log('desktop', 'iceState=', st);
      if (st === 'connected' || st === 'completed') {
        if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
        onStatus && onStatus('connected');
        onConnected && onConnected();
      } else if (st === 'disconnected') {
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
        // Kill the pc so the next incoming offer triggers ensurePc() fresh.
        // Leaving a failed pc in place means setRemoteDescription on the
        // replacement offer silently applies to a dead connection.
        log('desktop', 'ICE failed, resetting pc for next offer');
        onStatus && onStatus('reconnecting — restarting connection…');
        resetPc('ice-failed');
      }
    };
    return pc;
  };

  const handle = async (data) => {
    if (!data) return;
    if (data.offer) {
      // Every offer starts a fresh pc. Phone sends a new offer on every
      // reconnect path (first pair, ICE restart, rescan), and the prior pc
      // may be in have-local-answer or failed. Recreating avoids state
      // errors and guarantees ontrack fires for the current session.
      resetPc('new-offer');
      ensurePc();
      await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
      for (const c of pendingIce) { try { await pc.addIceCandidate(c); } catch {} }
      pendingIce = [];
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
    sock.addEventListener('open', () => {
      log('desktop', 'ws open, room=', roomId, 'peerId=', peerId);
      onStatus && onStatus('waiting for phone');
    });
    sock.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'signal' && msg.peer !== peerId) {
        handle(msg.data).catch((e) => log('handle err', e));
      } else if (msg.type === 'state') {
        // Skip state replays on a healthy pc. The server re-sends state on
        // every fresh WS open (including visibility-recovery reopens) and
        // replaying an old offer would tear down a working connection.
        if (isConnected()) { log('desktop', 'skip state: already connected'); return; }
        for (const d of extractFromState(msg.peers, peerId, 'phone')) {
          handle(d).catch((e) => log('handle err', e));
        }
      }
    };
    sock.onerror = (e) => { log('desktop', 'ws err', e); onStatus && onStatus('signaling error'); };
    sock.onclose = () => log('desktop', 'ws close');
  };
  wireWs(ws);

  const detachVis = installVisibilityRecovery({
    getWs: () => ws,
    setWs: (next) => { ws = next; wireWs(ws); },
    roomId, peerId,
    onMessage: (ev) => ws.onmessage(ev),
    onError:   (e)  => log('desktop', 'visibility-reopen err', e),
    onReconnect: () => onStatus && onStatus('signaling restored'),
  });

  return {
    stop: () => {
      detachVis();
      if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
      ws._close();
      resetPc('stop');
    },
    getState: () => ({
      role: 'desktop',
      peerId,
      roomId,
      signalingUrl: `${SIGNAL_URL}/tp-${roomId}/ws`,
      ws: ws ? ['CONNECTING','OPEN','CLOSING','CLOSED'][ws.readyState] : 'null',
      pc: pc ? {
        signalingState: pc.signalingState,
        iceConnectionState: pc.iceConnectionState,
        connectionState: pc.connectionState,
        iceGatheringState: pc.iceGatheringState,
      } : null,
      pendingIce: pendingIce.length,
    }),
    getStats: () => pc ? summariseStats(pc) : Promise.resolve(null),
  };
}

// ── Phone: dials desktop, sends video ────────────────────────
export async function startPhone(roomId, stream, callbacks = {}) {
  const { onStatus } = callbacks;
  const peerId = makePeerId('phone');
  let ws = openSignaling(roomId, peerId);
  const pc = new RTCPeerConnection({ iceServers: ICE });
  let pendingIce = [];
  let negotiating = false;
  let connectionTimer = null;
  let graceTimer = null;

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));
  pc.onicecandidate = (e) => { if (e.candidate) ws._send({ ice: e.candidate }); };
  pc.onsignalingstatechange  = () => log('phone', 'signalingState=', pc.signalingState);
  pc.onconnectionstatechange = () => log('phone', 'connectionState=', pc.connectionState);
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
      onStatus && onStatus('reconnecting — waiting on path…');
      if (!graceTimer) {
        graceTimer = setTimeout(() => {
          graceTimer = null;
          if (pc.iceConnectionState === 'disconnected') {
            log('phone', 'still disconnected after grace, restarting');
            try { pc.restartIce(); } catch {}
            renegotiate({ iceRestart: true });
          }
        }, DISCONNECT_GRACE_MS);
      }
    }
    if (st === 'failed') {
      log('phone', 'ICE failed, restarting');
      onStatus && onStatus('reconnecting — restarting connection…');
      try { pc.restartIce(); } catch {}
      renegotiate({ iceRestart: true });
    }
  };

  // Initial offer AND restart offers share this path. The negotiating flag
  // guards against overlap — previously the inline initial offer and the
  // disconnect-triggered restart could race the flag both directions.
  const renegotiate = async ({ iceRestart = false } = {}) => {
    if (negotiating) { log('phone', 'renegotiate skipped, in progress'); return; }
    negotiating = true;
    try {
      const offer = await pc.createOffer({ iceRestart });
      await pc.setLocalDescription(offer);
      ws._send({ offer });
    } catch (e) { log('phone', 'renegotiate err', e); }
    finally { negotiating = false; }
  };

  const isConnected = () => pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed';

  const handle = async (data) => {
    if (!data) return;
    if (data.answer) {
      log('phone', 'got answer');
      try {
        await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
        for (const c of pendingIce) { try { await pc.addIceCandidate(c); } catch {} }
        pendingIce = [];
      } catch (e) { log('phone', 'setRemoteDescription(answer) err', e); }
    } else if (data.ice) {
      const cand = new RTCIceCandidate(data.ice);
      if (pc.remoteDescription) { try { await pc.addIceCandidate(cand); } catch (e) { log('ice add err', e); } }
      else pendingIce.push(cand);
    }
  };

  const wireWs = (sock) => {
    sock.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'signal' && msg.peer !== peerId) {
        handle(msg.data).catch((e) => log('handle err', e));
      } else if (msg.type === 'state') {
        if (isConnected()) { log('phone', 'skip state: already connected'); return; }
        for (const d of extractFromState(msg.peers, peerId, 'desktop')) {
          handle(d).catch((e) => log('handle err', e));
        }
      }
    };
    sock.onerror = (e) => { log('phone', 'ws err', e); onStatus && onStatus('signaling error'); };
    sock.onclose = () => log('phone', 'ws close');
  };
  wireWs(ws);

  // Wait for WS open before sending. readyState check handles the race
  // where the socket opened before we attached the listener.
  await new Promise((res, rej) => {
    if (ws.readyState === WebSocket.OPEN) { log('phone', 'ws already open'); onStatus && onStatus('signal channel open'); return res(); }
    ws.addEventListener('open', () => {
      log('phone', 'ws open, peerId=', peerId);
      onStatus && onStatus('signal channel open');
      res();
    }, { once: true });
    setTimeout(() => rej(new Error('signaling timeout — close and reopen the desktop QR')), 10000);
  });

  // Initial offer goes through renegotiate() so the negotiating guard is
  // consistent with the restart path.
  await renegotiate({ iceRestart: false });
  onStatus && onStatus('offer sent, waiting for desktop…');

  connectionTimer = setTimeout(() => {
    if (pc.iceConnectionState === 'new' || pc.iceConnectionState === 'checking') {
      log('phone', 'connection timeout');
      onStatus && onStatus('connection timeout — close and rescan the desktop QR');
    }
  }, CONN_TIMEOUT_MS);

  const detachVis = installVisibilityRecovery({
    getWs: () => ws,
    setWs: (next) => { ws = next; wireWs(ws); },
    roomId, peerId,
    onMessage: (ev) => ws.onmessage(ev),
    onError:   (e)  => log('phone', 'visibility-reopen err', e),
    onReconnect: () => { onStatus && onStatus('signaling restored, restarting connection…'); renegotiate({ iceRestart: true }); },
  });

  return {
    stop: () => {
      detachVis();
      if (graceTimer)      { clearTimeout(graceTimer); graceTimer = null; }
      if (connectionTimer) { clearTimeout(connectionTimer); connectionTimer = null; }
      ws._close();
      pc.close();
      stream.getTracks().forEach((t) => t.stop());
    },
    getState: () => ({
      role: 'phone',
      peerId,
      roomId,
      signalingUrl: `${SIGNAL_URL}/tp-${roomId}/ws`,
      ws: ws ? ['CONNECTING','OPEN','CLOSING','CLOSED'][ws.readyState] : 'null',
      pc: {
        signalingState: pc.signalingState,
        iceConnectionState: pc.iceConnectionState,
        connectionState: pc.connectionState,
        iceGatheringState: pc.iceGatheringState,
      },
      pendingIce: pendingIce.length,
      negotiating,
    }),
    getStats: () => summariseStats(pc),
  };
}

export function makeRoomId() {
  const a = 'abcdefghjkmnpqrstuvwxyz23456789';
  let s = '';
  for (let i = 0; i < 6; i++) s += a[Math.floor(Math.random() * a.length)];
  return s;
}
