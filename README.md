# GoldWorm

GoldWorm is an experimental local cognitive engine and GUI runtime. The project is built around a Rust core, a localhost-only HTTP server, and a static browser UI served by Cargo through the `goldworn` binary.

The current implementation is intentionally local-first. The GUI is served from the repository, the backend binds to `127.0.0.1`, and the frontend avoids external CDN dependencies by default.

## Critical Model Paradigm

> ## ⚠️ GoldWorm starts completely untrained
>
> **The current situation:** The GoldWorm model is completely untrained "out-of-the-box".
>
> **The paradigm:** It has no pre-existing weights, prejudices, or answers.
>
> **Development:** The model grows, learns, and shapes its cognitive structure solely through direct interaction and user data. It is a symbiotic system.

This means GoldWorm should not be treated like a pretrained chatbot. Any useful behavior must emerge from local runtime state, explicit user interaction, local training workflows, and project data that the operator chooses to provide.

## What Is Included

- Rust backend runtime in the GoldWorm Cargo project.
- Local HTTP server exposed on `127.0.0.1:9090`.
- Static GUI served from `static/goldworm_gui.html`.
- Cargo start alias through the `goldworn` binary.
- Backend health endpoint for simulation monitoring.
- Defensive frontend rendering for chat output, neuron heatmap, and plot visualization.
- Zero-trust UI behavior with explicit fallbacks for missing or malformed data.

## Zero Trust Operating Rules

GoldWorm is designed to run under strict local trust boundaries:

- The server binds to `127.0.0.1`, not a public interface.
- The GUI is served locally by the Rust server.
- The frontend does not require external fonts, scripts, CDNs, or remote UI assets.
- The UI checks DOM elements before using them.
- The UI treats missing or malformed API data as expected failure input.
- The neuron heatmap and plot renderer show visible fallback output instead of silently breaking.
- Auto-scroll is disabled unless explicitly enabled by the operator.
- Simulation mode is shown clearly when the Rust backend is not verified.

## Requirements

Install these tools before running the project:

- Rust toolchain with Cargo.
- Git for version control.
- A local browser for opening the GUI.
- PowerShell on Windows, or a POSIX shell such as Bash on Linux.

The project does not require a public network listener for normal local GUI usage.

## Linux Installation

These commands assume a fresh Linux machine and a Bash-compatible terminal. GoldWorm still runs as a localhost-only service after installation.

### Debian or Ubuntu

```bash
sudo apt update
sudo apt install -y git curl build-essential pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
git clone https://github.com/loslos321-lab/GoldWorm.git
cd GoldWorm
cargo build --bin goldworn
cargo run goldworn
```

### Fedora

```bash
sudo dnf install -y git curl gcc gcc-c++ make pkgconf-pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
git clone https://github.com/loslos321-lab/GoldWorm.git
cd GoldWorm
cargo build --bin goldworn
cargo run goldworn
```

### Arch Linux

```bash
sudo pacman -Syu --needed git curl base-devel pkgconf
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
git clone https://github.com/loslos321-lab/GoldWorm.git
cd GoldWorm
cargo build --bin goldworn
cargo run goldworn
```

### Existing Linux Checkout

If the repository already exists locally, do not clone again. Use the existing directory:

```bash
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo build --bin goldworn
cargo run goldworn
```

When the server is ready, open this URL in a local browser:

```text
http://127.0.0.1:9090/gui
```

## Windows Installation

From PowerShell on an existing checkout:

```powershell
cd C:\Users\Student\GoldWorm
cargo build --bin goldworn
cargo run goldworn
```

## Start The Project

### Linux

Open a terminal and run:

```bash
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo run goldworn
```

### Windows

Open PowerShell and run:

```powershell
cd C:\Users\Student\GoldWorm
cargo run goldworn
```

When the server is ready, open:

```text
http://127.0.0.1:9090/gui
```

The health endpoint is:

```text
http://127.0.0.1:9090/health
```

A healthy Rust backend returns JSON with `backend` set to `rust` and `simulation` set to `false`.

## HTTP Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serves the GUI |
| `GET` | `/gui` | Serves the GUI |
| `GET` | `/health` | Reports backend status |
| `GET` | `/api/health` | Reports backend status |
| `POST` | `/api/send` | Sends a chat message to the engine |
| `POST` | `/api/clear` | Clears the current chat session history |
| `GET` | `/api/benchmark` | Runs the built-in benchmark endpoint |

Example chat request:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:9090/api/send" -Method POST -ContentType "application/json" -Body '{"message":"hello"}'
```

## GUI Modes

The GUI has three display modes:

- `MINI`: compact local operation with chat, controls, heatmap, and plot in a narrow layout.
- `CLASSIC`: chat plus controls and visualizations in a wider two-column layout.
- `MAX`: chat, controls, heatmap, and plot separated into a dense operator view.

The active tab uses a high-contrast green/cyan style. Inactive tabs remain dimmed and only brighten on hover.

## Frontend Stability

The GUI uses defensive state management for output and visualization:

- Chat input is stored before layout changes.
- Log output uses a strict `max-height` and `overflow-y: auto`.
- Scroll events are debounced.
- `scrollIntoView()` only runs when auto-scroll is enabled and a verified new message exists.
- Heatmap rendering checks the target container and validates numeric input.
- Plot rendering checks the target container and validates series input.
- If Plotly exists locally as `window.Plotly`, the GUI can use it.
- If Plotly is missing, the GUI renders a local SVG fallback.
- If data is null, undefined, or malformed, the GUI shows `Waiting for neuron signals...`.

## Training And Data Paradigm

GoldWorm is not shipped as a pretrained assistant. It has no default answer bank and no pretrained weights. Its cognitive behavior is expected to evolve through direct use, explicit local data, and training logic inside the project.

Operator guidance:

- Treat first-run output as untrained engine behavior.
- Do not assume benchmark output represents a trained model.
- Keep user data local unless you intentionally export it.
- Review generated artifacts before using them as training material.
- Keep training data, model artifacts, and secrets out of Git unless explicitly intended.

## Project Files

Important project paths:

| Path | Purpose |
| --- | --- |
| `Cargo.toml` | Cargo package metadata and dependencies |
| `src/bin/chat_server.rs` | Local HTTP server implementation |
| `src/bin/goldworn.rs` | Cargo start alias for the server |
| `static/goldworm_gui.html` | Local Zero Trust GUI |
| `src/` | Core Rust implementation |
| `tests/` | Rust tests and audits |
| `.gitignore` | Local ignore policy for build outputs and generated data |

## Development Commands

Build the local binary on Linux:

```bash
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo build --bin goldworn
```

Run the local GUI server on Linux:

```bash
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo run goldworn
```

Build and run on Windows:

```powershell
cd C:\Users\Student\GoldWorm
cargo build --bin goldworn
cargo run goldworn
```

Run tests:

```bash
cargo test
```

Check Git status:

```bash
git status --short --branch
```

## Troubleshooting

### Port 9090 is already in use

Stop any existing GoldWorm server process, then start again.

Linux:

```bash
pkill -f goldworn || true
pkill -f chat_server || true
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo run goldworn
```

Windows PowerShell:

```powershell
Stop-Process -Name goldworn,chat_server -Force
cd C:\Users\Student\GoldWorm
cargo run goldworn
```

### GUI reports simulation mode

Check whether the backend is running.

Linux:

```bash
curl -s http://127.0.0.1:9090/health
```

Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:9090/health" -UseBasicParsing
```

If the backend is not running, start it.

Linux:

```bash
cd /path/to/GoldWorm
. "$HOME/.cargo/env"
cargo run goldworn
```

Windows PowerShell:

```powershell
cd C:\Users\Student\GoldWorm
cargo run goldworn
```

### Heatmap or plot is waiting for signals

This is expected when signal data is unavailable, malformed, or intentionally static. The GUI renders a visible fallback instead of breaking the layout.

## Version Control Notes

Generated runtime logs should not be committed. Source code, documentation, tests, and intentional static assets should be reviewed before staging.

For this update, the semantic commit message is:

```text
feat(ui/docs): overhaul UI components and document untrained model paradigm in README
```
