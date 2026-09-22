#!/usr/bin/env node
/**
 * Cross-platform Cursor hook entry.
 * Buffers stdin and pipes it to the plugin venv Python (Windows inherit is unreliable).
 */
"use strict";

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const script = path.join(root, "scripts", "mlflow_cursor.py");

function exists(p) {
  try {
    return fs.existsSync(p);
  } catch (_) {
    return false;
  }
}

function resolvePython() {
  const venv = [
    path.join(root, ".venv", "Scripts", "python.exe"),
    path.join(root, ".venv", "bin", "python"),
    path.join(root, ".venv", "bin", "python3"),
  ];
  for (const p of venv) {
    if (exists(p)) return { cmd: p, args: [script] };
  }
  if (process.platform === "win32") {
    return { cmd: "py", args: ["-3", script] };
  }
  return { cmd: "python3", args: [script] };
}

function readStdin() {
  return new Promise((resolve, reject) => {
    const chunks = [];
    process.stdin.on("data", (c) => chunks.push(c));
    process.stdin.on("end", () => resolve(Buffer.concat(chunks)));
    process.stdin.on("error", reject);
    // Cursor sometimes closes stdin immediately with empty body.
    if (process.stdin.readableEnded) {
      resolve(Buffer.concat(chunks));
    }
  });
}

function failOpen(message) {
  try {
    process.stderr.write(message + "\n");
  } catch (_) {
    /* ignore */
  }
  process.stdout.write("{}\n");
  process.exit(0);
}

(async () => {
  let input;
  try {
    input = await readStdin();
  } catch (err) {
    failOpen(`run_hook.js: stdin read failed: ${err.message}`);
    return;
  }

  const { cmd, args } = resolvePython();
  const child = spawn(cmd, args, {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: process.env.CURSOR_PROJECT_DIR || process.cwd(),
    env: {
      ...process.env,
      MLFLOW_DISABLE_AGENT_HINT: process.env.MLFLOW_DISABLE_AGENT_HINT || "1",
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
    },
    windowsHide: true,
  });

  let stdout = Buffer.alloc(0);
  let stderr = Buffer.alloc(0);
  child.stdout.on("data", (c) => {
    stdout = Buffer.concat([stdout, c]);
  });
  child.stderr.on("data", (c) => {
    stderr = Buffer.concat([stderr, c]);
  });

  child.on("error", (err) => {
    failOpen(`run_hook.js: failed to start ${cmd}: ${err.message}`);
  });

  child.on("exit", (code) => {
    if (stderr.length) {
      try {
        process.stderr.write(stderr);
      } catch (_) {
        /* ignore */
      }
    }
    const text = stdout.toString("utf8").trim();
    if (text) {
      process.stdout.write(text + (text.endsWith("\n") ? "" : "\n"));
    } else {
      process.stdout.write("{}\n");
    }
    process.exit(code == null ? 0 : code);
  });

  child.stdin.write(input);
  child.stdin.end();
})();
