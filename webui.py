#!/usr/bin/env python3
"""
WormBrain 3D Trajectory Dashboard — purple neon retrofuture Gradio UI.

Communicates with the Rust chat_server HTTP API running on port 9090.
Displays live 3D token trajectories through the 16-D semantic manifold.

Usage:
    python3 webui.py [--port 7860] [--api-port 9090] [--share]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
VOCAB_PATH = REPO_ROOT / "static_vocabulary.txt"
MANIFOLD_DIM = 16
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# ── Token → Coordinate (replicates Rust geometry.rs token_to_coord) ──


def token_hash_coord(token: str) -> np.ndarray:
    """Deterministic hash → unit-length 16-D coordinate (Rust token_hash_coord)."""
    raw = token.encode("utf-8")
    arr = np.zeros(MANIFOLD_DIM, dtype=np.float64)
    for d in range(MANIFOLD_DIM):
        v = 0.0
        for k, b in enumerate(raw):
            seed = d * 37 + k * 13
            v += float(b) * math.sin(float(seed))
        arr[d] = v
    norm = np.sqrt(arr @ arr)
    return arr / norm if norm > 1e-15 else arr


def token_to_coord_py(token: str) -> np.ndarray:
    """Replicates Rust char_ngram_to_coord + token_hash_coord fallback."""
    lower = token.lower()
    lower_bytes = lower.encode("utf-8")
    acc = np.zeros(MANIFOLD_DIM, dtype=np.float64)
    count = 0
    for nlen in (3, 4):
        if len(lower_bytes) >= nlen:
            for start in range(len(lower_bytes) - nlen + 1):
                ngram = lower_bytes[start:start + nlen].decode("ascii", errors="replace")
                acc += token_hash_coord(ngram)
                count += 1
    if count == 0:
        return token_hash_coord(token)
    norm = np.sqrt(acc @ acc)
    return acc / norm if norm > 1e-15 else token_hash_coord(token)


# ── Vocabulary Database ──


class VocabularyDB:
    """Load static_vocabulary.txt, compute 16-D coords, PCA→3D."""

    def __init__(self, path: str = str(VOCAB_PATH)):
        self.tokens: list[str] = []
        self.coords_16d: np.ndarray | None = None
        self.coords_3d: np.ndarray | None = None
        self.pca: object | None = None
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            print(f"  WARNING: vocabulary not found: {path}", file=sys.stderr)
            return
        with open(path) as f:
            self.tokens = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        if not self.tokens:
            return
        self.coords_16d = np.array([token_to_coord_py(t) for t in self.tokens], dtype=np.float64)
        self._compute_pca()
        print(f"  Loaded {len(self.tokens)} vocabulary entries → PCA 3D", file=sys.stderr)

    def _compute_pca(self) -> None:
        from sklearn.decomposition import PCA
        if self.coords_16d is None or self.coords_16d.shape[0] < 3:
            return
        self.pca = PCA(n_components=3)
        self.coords_3d = self.pca.fit_transform(self.coords_16d)

    def lookup_3d(self, token: str) -> list[float]:
        if self.coords_3d is None or self.pca is None:
            return [0.0, 0.0, 0.0]
        c16 = token_to_coord_py(token)
        return self.pca.transform(c16.reshape(1, -1))[0].tolist()


# ── WormBrain Bridge (HTTP client to Rust chat_server) ──


class WormBrainBridge:
    """Communicates with the Rust chat_server HTTP API."""

    def __init__(self, api_base: str = "http://127.0.0.1:9090"):
        self.api_base = api_base.rstrip("/")

    def send_message(self, text: str) -> str:
        data = json.dumps({"message": text}).encode()
        req = urllib.request.Request(
            f"{self.api_base}/api/send",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()).get("reply", "")

    def clear_history(self) -> None:
        req = urllib.request.Request(
            f"{self.api_base}/api/clear",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)

    def run_benchmark(self) -> dict:
        with urllib.request.urlopen(f"{self.api_base}/api/benchmark", timeout=120) as resp:
            return json.loads(resp.read().decode()).get("results", {})


# ── Plotly.js HTML Generator ──


def _gen_trajectory_html(
    xs: list[float], ys: list[float], zs: list[float],
    token_labels: list[str],
    vocab_3d: np.ndarray | None,
    width: str = "100%", height: str = "440px",
) -> str:
    vocab_trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": vocab_3d[:, 0].tolist() if vocab_3d is not None else [],
        "y": vocab_3d[:, 1].tolist() if vocab_3d is not None else [],
        "z": vocab_3d[:, 2].tolist() if vocab_3d is not None else [],
        "marker": {"size": 1.5, "color": "rgba(140,90,220,0.12)", "symbol": "circle"},
        "hoverinfo": "none",
        "showlegend": False,
    }
    traj_line = {
        "type": "scatter3d",
        "mode": "lines+markers",
        "x": xs,
        "y": ys,
        "z": zs,
        "line": {"color": "rgba(180,60,255,0.85)", "width": 3},
        "marker": {
            "size": 5,
            "color": "rgba(255,60,200,1)",
            "symbol": "diamond",
            "line": {"color": "rgba(255,255,255,0.3)", "width": 1},
        },
        "text": token_labels,
        "hoverinfo": "text",
        "showlegend": False,
    }
    graph_json = json.dumps({
        "data": [vocab_trace, traj_line],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "scene": {
                "bgcolor": "rgba(0,0,0,0)",
                "xaxis": {"showgrid": False, "showticklabels": False, "title": "", "zeroline": False},
                "yaxis": {"showgrid": False, "showticklabels": False, "title": "", "zeroline": False},
                "zaxis": {"showgrid": False, "showticklabels": False, "title": "", "zeroline": False},
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}, "up": {"x": 0, "y": 0, "z": 1}},
            },
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "autosize": True,
        },
    })
    return f"""<div id="traj-plot" style="width:{width};height:{height};"></div>
<script>
(function(){{
var el = document.getElementById('traj-plot');
if(!el) return;
var d = {graph_json};
if(typeof Plotly === 'undefined'){{
var s=document.createElement('script');
s.src='{PLOTLY_CDN}';
s.onload=function(){{Plotly.newPlot('traj-plot',d.data,d.layout,{{responsive:true,displayModeBar:false}});}};
document.head.appendChild(s);
}}else{{Plotly.react('traj-plot',d.data,d.layout,{{responsive:true,displayModeBar:false}});}}
}})();
</script>"""


def build_trajectory_data(reply: str, vocab: VocabularyDB) -> tuple[list[float], list[float], list[float], list[str]]:
    xs, ys, zs, labels = [], [], [], []
    for word in reply.split():
        c3 = vocab.lookup_3d(word)
        xs.append(c3[0])
        ys.append(c3[1])
        zs.append(c3[2])
        labels.append(word)
    return xs, ys, zs, labels


# ── Telemetry JSON ──


def telemetry_json(reply: str, vocab: VocabularyDB) -> str:
    items = []
    for word in reply.split():
        c3 = vocab.lookup_3d(word)
        items.append({
            "token": word,
            "x": round(c3[0], 4),
            "y": round(c3[1], 4),
            "z": round(c3[2], 4),
        })
    return json.dumps(items, indent=2) if items else json.dumps([], indent=2)


# ── Server Process Manager ──


def find_server_binary() -> Path:
    candidates = [
        REPO_ROOT / "target" / "release" / "chat_server",
        REPO_ROOT / "target" / "debug" / "chat_server",
    ]
    for c in candidates:
        if c.exists():
            return c
    print("=" * 60, file=sys.stderr)
    print("  Chat server binary not found.", file=sys.stderr)
    print(file=sys.stderr)
    print("  Build it:", file=sys.stderr)
    print(f"    cd {REPO_ROOT} && cargo build --release --bin chat_server", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    raise SystemExit(1)


class ServerProcess:
    """Launches and manages the Rust chat_server subprocess."""

    def __init__(self, port: int, server_bin: Path):
        self.port = port
        self.api_base = f"http://127.0.0.1:{port}"
        self._server_bin = server_bin
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        print(f"  Starting chat server on port {self.port} ...", file=sys.stderr)
        self._process = subprocess.Popen(
            [str(self._server_bin), f"--port={self.port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{self.api_base}/", timeout=2):
                    print("  Chat server ready.", file=sys.stderr)
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        stderr = b""
        if self._process and self._process.stderr:
            stderr = self._process.stderr.read()
        raise RuntimeError(
            f"Chat server failed to start within 15s.\nStderr: {stderr.decode(errors='replace')}"
        )

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


# ── Neon RetroFuture CSS ──


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --neon-purple: #b43cff;
  --neon-magenta: #ff3cc8;
  --neon-cyan: #3cffd0;
  --bg-deep: #0a0015;
  --bg-panel: #12001e;
  --bg-card: #1a0a2e;
  --text-primary: #e0d0f0;
  --text-dim: #8070a0;
  --border-glow: rgba(180,60,255,0.25);
}

body { background: var(--bg-deep); font-family: 'JetBrains Mono', monospace; }

body::after {
  content: ''; position: fixed; inset: 0;
  background: repeating-linear-gradient(0deg, transparent 0px, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px);
  pointer-events: none; z-index: 9999;
}

.gradio-container { background: transparent !important; max-width: 100% !important; padding: 0 !important; }

/* chat bubbles */
.message { border-radius: 6px !important; margin: 4px 0 !important; }
.bot-message { background: rgba(26,10,46,0.85) !important; border-left: 3px solid var(--neon-purple) !important; color: var(--text-primary) !important; }
.user-message { background: rgba(40,40,140,0.15) !important; border-right: 3px solid var(--neon-cyan) !important; color: var(--text-primary) !important; }

/* inputs */
input, textarea, .gr-text-input {
  background: var(--bg-panel) !important; border: 1px solid var(--border-glow) !important;
  color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace !important; border-radius: 6px !important;
}
input:focus, textarea:focus { border-color: var(--neon-purple) !important; box-shadow: 0 0 12px rgba(180,60,255,0.3) !important; }

/* buttons */
button, .gr-button {
  font-family: 'Orbitron', sans-serif !important; font-weight: 700 !important;
  letter-spacing: 1px !important; text-transform: uppercase !important;
  border: 1px solid var(--neon-purple) !important;
  background: linear-gradient(135deg, rgba(180,60,255,0.2), rgba(255,60,200,0.1)) !important;
  color: var(--neon-purple) !important; border-radius: 6px !important; transition: all .2s ease !important;
}
button:hover, .gr-button:hover {
  background: linear-gradient(135deg, rgba(180,60,255,0.4), rgba(255,60,200,0.2)) !important;
  box-shadow: 0 0 20px rgba(180,60,255,0.4) !important; color: #fff !important;
}

/* panels */
.gr-box, .gr-form, .panel { background: var(--bg-panel) !important; border: 1px solid var(--border-glow) !important; border-radius: 8px !important; }

/* telemetry */
#telemetry-box textarea {
  font-family: 'JetBrains Mono', monospace !important; font-size: .7rem !important;
  color: #b0a0d0 !important; background: rgba(0,0,0,0.3) !important;
  border: 1px solid rgba(180,60,255,0.15) !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(180,60,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(180,60,255,0.5); }

/* system banner shimmer */
@keyframes shimmer { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
"""


# ── Gradio App ──


def main() -> None:
    parser = argparse.ArgumentParser(description="WormBrain 3D Trajectory Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Gradio UI port")
    parser.add_argument("--api-port", type=int, default=9090, help="Rust server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio not found. Install: pip install gradio", file=sys.stderr)
        sys.exit(2)

    server_bin = find_server_binary()
    server = ServerProcess(port=args.api_port, server_bin=server_bin)
    server_started = False
    try:
        server.start()
        server_started = True
    except RuntimeError as e:
        print(f"  WARNING: {e}", file=sys.stderr)

    bridge = WormBrainBridge(api_base=f"http://127.0.0.1:{args.api_port}")
    vocab = VocabularyDB()

    if vocab.coords_3d is None:
        print("  WARNING: vocabulary not loaded; 3D trajectory will be empty.", file=sys.stderr)

    def respond(message: str, history: list):
        if not message.strip():
            empty_html = _gen_trajectory_html([], [], [], [], vocab.coords_3d)
            return history, empty_html, json.dumps([], indent=2), ""
        reply = bridge.send_message(message)
        xs, ys, zs, labels = build_trajectory_data(reply, vocab)
        html = _gen_trajectory_html(xs, ys, zs, labels, vocab.coords_3d)
        tele = telemetry_json(reply, vocab)
        history = list(history) if history else []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, html, tele, ""

    def clear_all():
        bridge.clear_history()
        empty_html = _gen_trajectory_html([], [], [], [], vocab.coords_3d)
        return [], empty_html, json.dumps([], indent=2)

    def run_benchmark():
        results = bridge.run_benchmark()
        lines = []
        for i, (name, score) in enumerate(results.get("rankings", []), 1):
            lines.append(f"{i}. {name}: {score:.4f}")
        details = results.get("scores", {})
        for name in details:
            s = details[name]
            parts = []
            for k in ("accuracy", "coherence", "creativity"):
                parts.append(f"{k}={s.get(k, 0):.3f}")
            lines.append(f"  {name}: {', '.join(parts)}")
        return "\n".join(lines) if lines else "No benchmark data."

    with gr.Blocks(
        title="WormBrain Trajectory Dashboard",
        css=CSS,
        fill_height=True,
    ) as demo:
        gr.HTML("""
<div style="
  background:linear-gradient(135deg,#1a0a2e 0%,#2d1b4e 50%,#1a0a2e 100%);
  border-bottom:2px solid rgba(180,60,255,0.4);
  padding:10px 24px; text-align:center;
  box-shadow:0 0 20px rgba(180,60,255,0.15);
">
  <div style="
    font-family:'Orbitron',sans-serif; font-size:1.5rem; font-weight:700;
    letter-spacing:4px; text-transform:uppercase;
    background:linear-gradient(90deg,#b43cff,#ff3cc8,#b43cff);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    animation:shimmer 3s linear infinite;
    text-shadow:0 0 30px rgba(180,60,255,0.5);
  ">⊗ WormBrain Trajectory Dashboard ⊗</div>
  <div style="
    font-family:'JetBrains Mono',monospace; font-size:.75rem;
    color:rgba(180,180,255,0.6); letter-spacing:2px; margin-top:4px;
  ">302-NEURON C. ELEGANS CONNECTOME · NON-LINEAR GEOMETRIC ROUTING · 16-D SEMANTIC MANIFOLD</div>
</div>""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=380):
                chatbot = gr.Chatbot(
                    label=None,
                    type="messages",
                    height=440,
                    bubble_full_width=False,
                    show_label=False,
                    container=True,
                )
                msg = gr.Textbox(
                    label=None, placeholder="Type a message for the worm…",
                    show_label=False, container=False,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
                    bench_btn = gr.Button("BM", scale=1)

            with gr.Column(scale=5, min_width=480):
                traj_html = gr.HTML(
                    value=_gen_trajectory_html([], [], [], [], vocab.coords_3d),
                    show_label=False,
                )
                with gr.Row():
                    tele_box = gr.Textbox(
                        label="Telemetry",
                        value=json.dumps([], indent=2),
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        elem_id="telemetry-box",
                        scale=3,
                    )
                    bench_out = gr.Textbox(
                        label="Benchmark",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        scale=2,
                    )

        send_btn.click(respond, [msg, chatbot], [chatbot, traj_html, tele_box, msg], queue=True)
        msg.submit(respond, [msg, chatbot], [chatbot, traj_html, tele_box, msg], queue=True)
        clear_btn.click(clear_all, outputs=[chatbot, traj_html, tele_box], queue=True)
        bench_btn.click(run_benchmark, outputs=[bench_out], queue=True)

    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
