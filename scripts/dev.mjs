// One command to run SignalLens locally: `npm start` from the repo root.
//
// First run: creates backend/.venv and installs Python + npm dependencies.
// Every run: starts the API (uvicorn, auto-reload, :8000) and the frontend
// (Vite, :5173) side by side with labelled output. Ctrl+C stops both.
// Uses only Node built-ins, so the root folder needs no `npm install`.

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";
const backend = join(root, "backend");
const frontend = join(root, "frontend");
const venvPython = join(backend, ".venv", isWin ? "Scripts" : "bin", isWin ? "python.exe" : "python");

// npm is a .cmd script on Windows, which Node can only start through a shell;
// a shell gets the whole command as one string (passing an args array with
// shell: true is deprecated because the args aren't escaped).
function launch(cmd, args, options, sync = false) {
  const viaShell = isWin && cmd === "npm";
  const fn = sync ? spawnSync : spawn;
  return viaShell ? fn([cmd, ...args].join(" "), { ...options, shell: true }) : fn(cmd, args, options);
}

function setupStep(label, cmd, args, cwd) {
  console.log(`\n[setup] ${label}`);
  const r = launch(cmd, args, { cwd, stdio: "inherit" }, true);
  if (r.error || r.status !== 0) {
    console.error(`[setup] failed: ${label}${r.error ? ` (${r.error.message})` : ""}`);
    process.exit(r.status || 1);
  }
}

if (!existsSync(venvPython)) {
  setupStep("creating Python virtual environment (backend/.venv)", isWin ? "python" : "python3", ["-m", "venv", ".venv"], backend);
  setupStep("installing Python packages", venvPython, ["-m", "pip", "install", "-r", "requirements-dev.txt"], backend);
}
if (!existsSync(join(frontend, "node_modules"))) {
  setupStep("installing frontend packages", "npm", ["install"], frontend);
}

const services = [
  { name: "api", colour: "\x1b[36m", cmd: venvPython, args: ["-m", "uvicorn", "signallens.api.app:app", "--reload", "--port", "8000"], cwd: backend },
  { name: "web", colour: "\x1b[35m", cmd: "npm", args: ["run", "dev"], cwd: frontend },
];

const children = [];
let stopping = false;

function prefix(name, colour, stream, target) {
  let buffer = "";
  stream.on("data", (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop();
    for (const line of lines) target.write(`${colour}[${name}]\x1b[0m ${line}\n`);
  });
}

function stopAll(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode !== null) continue;
    // On Windows, killing the shell leaves Vite/uvicorn running; kill the whole tree.
    if (isWin) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    else child.kill("SIGINT");
  }
  setTimeout(() => process.exit(code), 300);
}

for (const s of services) {
  const child = launch(s.cmd, s.args, { cwd: s.cwd, env: { ...process.env, FORCE_COLOR: "1" } });
  prefix(s.name, s.colour, child.stdout, process.stdout);
  prefix(s.name, s.colour, child.stderr, process.stderr);
  child.on("exit", (code) => {
    if (!stopping) {
      console.error(`\n[${s.name}] stopped (exit code ${code}). Shutting down the other server.`);
      stopAll(code || 1);
    }
  });
  children.push(child);
}

console.log("\n  SignalLens is starting…");
console.log("  App:  http://localhost:5173   (open this)");
console.log("  API:  http://localhost:8000/api/health");
console.log("  Press Ctrl+C to stop both.\n");

process.on("SIGINT", () => stopAll(0));
process.on("SIGTERM", () => stopAll(0));
