#!/usr/bin/env python3
"""
trinet_bridge.py  -  Local MJPEG bridge for the Sentrix Trinet UVC RGB camera.

WHY THIS EXISTS
---------------
The Trinet camera (Rockchip 2207:0016) outputs H.264 over UVC.
Chrome's getUserMedia cannot open it (format negotiation fails in the MF
pipeline).  This script opens the camera via OpenCV's MSMF backend, decodes
each frame, and serves a standard multipart/x-mixed-replace MJPEG stream that
the React dashboard displays in a plain <img> element.

MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED (0xC00D3EA2 / -1072873822)
-------------------------------------------------------------------
The Trinet hardware re-enumerates its Media Foundation stream roughly 6 s
after any prior connection closes (a normal documented Rockchip behavior, not
a fault).  If a Chrome getUserMedia attempt just failed, or the bridge was
recently stopped, the device will be in this re-enumeration window.
The bridge detects consecutive grab failures, waits 8 s, and retries - which
is the same strategy the SDK's TrinetWatcher uses with its recovery_delay.

QUICK START
-----------
  pip install opencv-python

  # Find the device index for the Trinet:
  python trinet_bridge.py --list

  # Start the bridge (Trinet is almost always MSMF index 0):
  python trinet_bridge.py --device 0 --backend msmf

  # In the Dashboard -> EGOCENTRIC CAM -> click BRIDGE -> CONNECT
  # URL: http://localhost:8080/stream.mjpeg

OPTIONS
-------
  --device  N        Camera index (default 0)
  --backend msmf|dshow   Capture backend (default: msmf, needed for Trinet)
  --port    N        HTTP port (default 8080)
  --quality 1-100    JPEG quality (default 80)
  --scale   0.1-1.0  Downscale factor, e.g. 0.5 = 960x540 (default 1.0)
  --width / --height / --fps  Requested capture format (default 1920 1080 30)
  --list             Print available cameras on both backends and exit
"""

import argparse
import contextlib
import http.server
import os
import signal
import socketserver
import sys
import threading
import time

# ── dependency check ──────────────────────────────────────────────────────────

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed.")
    print("  pip install opencv-python")
    sys.exit(1)

# ── constants ─────────────────────────────────────────────────────────────────

BACKENDS = {
    'msmf':  cv2.CAP_MSMF,    # Media Foundation  — finds the Trinet
    'dshow': cv2.CAP_DSHOW,   # DirectShow        — finds integrated / OBS
}

# How many consecutive empty frames trigger a reopen attempt
FAIL_STREAK_THRESHOLD = 8

# Seconds to wait before retrying after MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED
# The camera takes ~6 s to re-enumerate; 8 s gives a comfortable margin.
REOPEN_WAIT_S = 8

# Maximum reopen attempts before giving up
MAX_REOPENS = 15

BOUNDARY = b'--trinetframe'

# ── shared frame store ────────────────────────────────────────────────────────

class FrameStore:
    """Thread-safe latest-frame store."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None

    def put(self, jpeg: bytes):
        with self._lock:
            self._frame = jpeg

    def get(self):
        with self._lock:
            return self._frame


_store   = FrameStore()
_running = True

# ── camera helpers ────────────────────────────────────────────────────────────

def _open(index: int, backend_key: str):
    """Open one camera index via the given backend; suppress OpenCV warnings."""
    backend = BACKENDS.get(backend_key, cv2.CAP_MSMF)
    devnull = open(os.devnull, 'w')
    with contextlib.redirect_stderr(devnull):
        cap = cv2.VideoCapture(index, backend)
    devnull.close()
    if cap.isOpened():
        return cap
    cap.release()
    return None


# ── capture thread ────────────────────────────────────────────────────────────

def capture_loop(device_index: int, backend_key: str,
                 width: int, height: int, fps: int,
                 quality: int, scale: float):
    """
    Outer loop: open camera, stream frames, detect invalidation, wait, reopen.
    Mirrors the recovery strategy in the SDK's TrinetWatcher.
    """
    global _running
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    reopens = 0

    while _running:
        # ── open (or reopen) ─────────────────────────────────────────────────
        print(f"[bridge] Opening camera index {device_index} "
              f"via {backend_key.upper()} ...")
        cap = _open(device_index, backend_key)

        if cap is None and backend_key == 'msmf':
            print("[bridge] MSMF failed — trying DirectShow fallback ...")
            cap = _open(device_index, 'dshow')

        if cap is None:
            print(f"[bridge] ERROR: Cannot open camera index {device_index}. "
                  "Run --list to check indices.")
            _running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = cap.get(cv2.CAP_PROP_FPS)
        bk = cap.getBackendName() if hasattr(cap, 'getBackendName') else backend_key.upper()
        print(f"[bridge] Streaming via {bk}: {aw}x{ah} @ {af:.0f} fps")

        # ── inner frame loop ─────────────────────────────────────────────────
        fail_streak = 0

        while _running:
            ret, frame = cap.read()

            if not ret:
                fail_streak += 1
                time.sleep(0.05)
                if fail_streak >= FAIL_STREAK_THRESHOLD:
                    # Consecutive failures — camera stream has been invalidated.
                    break
                continue

            fail_streak = 0
            reopens     = 0   # successful frame resets the reopen counter

            if scale != 1.0:
                nw = max(1, int(aw * scale))
                nh = max(1, int(ah * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

            ok, buf = cv2.imencode('.jpg', frame, encode_params)
            if ok:
                _store.put(bytes(buf))

        cap.release()

        if not _running:
            break

        # ── recovery: wait for re-enumeration ────────────────────────────────
        reopens += 1
        if reopens > MAX_REOPENS:
            print(f"[bridge] ERROR: Failed to recover after {MAX_REOPENS} attempts.")
            _running = False
            break

        print(f"[bridge] Stream lost (MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED). "
              f"Attempt {reopens}/{MAX_REOPENS}.")
        print(f"[bridge] Trinet re-enumerates in ~6 s — waiting {REOPEN_WAIT_S} s ...")

        t0 = time.time()
        while _running and (time.time() - t0) < REOPEN_WAIT_S:
            time.sleep(0.25)

    print("[bridge] Capture thread stopped.")


# ── HTTP server ───────────────────────────────────────────────────────────────

class MJPEGHandler(http.server.BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ('/stream.mjpeg', '/stream'):
            self._stream()
        elif self.path in ('/health', '/'):
            self._health()
        else:
            self.send_error(404)

    def _health(self):
        body = b'Trinet MJPEG bridge OK'
        self.send_response(200)
        self.send_header('Content-Type',   'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         f'multipart/x-mixed-replace; '
                         f'boundary={BOUNDARY.decode()}')
        self.send_header('Cache-Control', 'no-cache')
        self._cors()
        self.end_headers()

        while _running:
            jpeg = _store.get()
            if jpeg is None:
                time.sleep(0.01)
                continue
            try:
                self.wfile.write(
                    BOUNDARY + b'\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n'
                    b'\r\n' + jpeg + b'\r\n'
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                break

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ('200', '204'):
            print(f"[http] {fmt % args}")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads    = True
    allow_reuse_address = True


# ── device listing ────────────────────────────────────────────────────────────

def list_devices():
    for bk, bid in BACKENDS.items():
        hint = ("MSMF  <-- use this for Trinet / H.264 UVC"
                if bk == 'msmf'
                else "DSHOW <-- integrated webcam / OBS virtual camera")
        print(f"\n[{bk.upper()}]  {hint}")
        found = False
        for i in range(8):
            cap = _open(i, bk)
            if cap:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f = cap.get(cv2.CAP_PROP_FPS)
                tag = "  <-- likely Trinet" if w >= 1920 and bk == 'msmf' else ""
                print(f"  index {i}:  {w}x{h} @ {f:.0f} fps{tag}")
                cap.release()
                found = True
        if not found:
            print("  (none found)")
    print()
    print("Trinet command:  python trinet_bridge.py --device 0 --backend msmf")
    print("Other cameras:   python trinet_bridge.py --device N --backend dshow")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Sentrix Trinet camera -> MJPEG HTTP bridge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--list',    action='store_true',
                    help='List cameras on all backends and exit')
    ap.add_argument('--device',  type=int,   default=0,
                    help='Camera index (default 0)')
    ap.add_argument('--backend', choices=['msmf', 'dshow'], default='msmf',
                    help='Capture backend: msmf (default) or dshow')
    ap.add_argument('--port',    type=int,   default=8080,
                    help='HTTP port (default 8080)')
    ap.add_argument('--quality', type=int,   default=80,
                    help='JPEG quality 1-100 (default 80)')
    ap.add_argument('--scale',   type=float, default=1.0,
                    help='Downscale factor, e.g. 0.5 = half resolution')
    ap.add_argument('--width',   type=int,   default=1920)
    ap.add_argument('--height',  type=int,   default=1080)
    ap.add_argument('--fps',     type=int,   default=30)
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    global _running

    ct = threading.Thread(
        target=capture_loop,
        args=(args.device, args.backend,
              args.width, args.height, args.fps,
              args.quality, args.scale),
        daemon=True,
    )
    ct.start()

    # Give the camera time to open (and survive the first re-enumeration window)
    # before the HTTP server starts accepting connections.
    time.sleep(2)

    if not _running:
        sys.exit(1)

    srv = Server(('0.0.0.0', args.port), MJPEGHandler)
    print(f"[bridge] Stream:  http://localhost:{args.port}/stream.mjpeg")
    print(f"[bridge] Health:  http://localhost:{args.port}/health")
    print(f"[bridge] Ctrl-C to stop.\n")

    def shutdown(sig, frame):
        global _running
        print("\n[bridge] Shutting down ...")
        _running = False
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

    _running = False
    ct.join(timeout=5)
    print("[bridge] Done.")


if __name__ == '__main__':
    main()
