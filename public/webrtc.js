// Minimal WebRTC video-track pairing via signal.neevs.io.
// Desktop waits for a phone to dial in. Phone sends camera video.
// Protocol matches catwatcher: {type:'signal', peer:<role>, data:{offer|answer|ice}}

const SIGNAL_URL = 'wss://signal.neevs.io';
const ICE = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun.cloudflare.com:3478' },
];

function openSignaling(roomId, role) {
  const ws = new WebSocket(`${SIGNAL_URL}/tp-${roomId}/ws`);
  const heartbeat = setInterval(() => {
    if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' }));
  }, 20000);
  ws._heartbeat = heartbeat;
  ws._role = role;
  ws._send = (data) => {
    if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'signal', peer: role, data }));
  };
  ws._close = () => {
    clearInterval(heartbeat);
    try { ws.close(); } catch {}
  };
  return ws;
}

// Desktop: waits for phone to dial in. Calls onRemoteStream(stream) when video arrives.
export function startDesktop(roomId, callbacks = {}) {
  const { onRemoteStream, onStatus, onConnected } = callbacks;
  const ws = openSignaling(roomId, 'desktop');
  let pc = null;
  const pendingIce = [];

  const ensurePc = () => {
    if (pc) return pc;
    pc = new RTCPeerConnection({ iceServers: ICE });
    pc.onicecandidate = (e) => e.candidate && ws._send({ ice: e.candidate });
    pc.ontrack = (e) => onRemoteStream && onRemoteStream(e.streams[0]);
    pc.oniceconnectionstatechange = () => {
      const st = pc.iceConnectionState;
      onStatus && onStatus(st);
      if (st === 'connected' || st === 'completed') onConnected && onConnected();
    };
    return pc;
  };

  ws.onopen = () => onStatus && onStatus('waiting for phone');
  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== 'signal' || msg.peer === 'desktop') return;
    const data = msg.data || {};
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
      if (pc && pc.remoteDescription) { try { await pc.addIceCandidate(cand); } catch {} }
      else pendingIce.push(cand);
    }
  };
  ws.onerror = () => onStatus && onStatus('signaling error');
  ws.onclose = () => onStatus && onStatus('signaling closed');

  return {
    stop: () => { ws._close(); if (pc) pc.close(); pc = null; },
  };
}

// Phone: dials the desktop, sends video track. Returns a handle with .stop().
export async function startPhone(roomId, stream, callbacks = {}) {
  const { onStatus } = callbacks;
  const ws = openSignaling(roomId, 'phone');
  const pc = new RTCPeerConnection({ iceServers: ICE });
  const pendingIce = [];

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));
  pc.onicecandidate = (e) => e.candidate && ws._send({ ice: e.candidate });
  pc.oniceconnectionstatechange = () => onStatus && onStatus(pc.iceConnectionState);

  await new Promise((res) => { ws.onopen = res; });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws._send({ offer });
  onStatus && onStatus('offer sent');

  ws.onmessage = async (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== 'signal' || msg.peer === 'phone') return;
    const data = msg.data || {};
    if (data.answer) {
      await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
      for (const c of pendingIce) { try { await pc.addIceCandidate(c); } catch {} }
      pendingIce.length = 0;
    } else if (data.ice) {
      const cand = new RTCIceCandidate(data.ice);
      if (pc.remoteDescription) { try { await pc.addIceCandidate(cand); } catch {} }
      else pendingIce.push(cand);
    }
  };

  return {
    stop: () => { ws._close(); pc.close(); stream.getTracks().forEach((t) => t.stop()); },
  };
}

export function makeRoomId() {
  const a = 'abcdefghjkmnpqrstuvwxyz23456789';
  let s = '';
  for (let i = 0; i < 6; i++) s += a[Math.floor(Math.random() * a.length)];
  return s;
}
