// Record a demo as PNG frames by driving Chrome over the DevTools protocol.
// usage: node tools/record_demo.mjs <file-url> <out-dir> <n-frames> <game-steps-per-frame>
// Chrome must be running with --remote-debugging-port=9333 (tools/make_videos.py starts it).
// The game is advanced deterministically with the page's own "Step" button, so the video
// shows exactly the simulated game and does not depend on machine speed.
import fs from "node:fs";

const [, , url, outDir, nFrames, stepsPerFrame] = process.argv;
const list = await (await fetch("http://127.0.0.1:9333/json")).json();
const tab = list.find((t) => t.type === "page");
const ws = new WebSocket(tab.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ev = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result.result.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1100, height: 860, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
await sleep(1500);
// clip = the two cards (game + brain panel), measured from the page
const box = await ev(`(() => { const g = document.querySelector(".grid").getBoundingClientRect();
  const h = document.querySelector("h1").getBoundingClientRect();
  return { x: Math.floor(g.left) - 8, y: Math.floor(h.top) - 8, width: Math.ceil(g.width) + 16, height: Math.ceil(g.bottom - h.top) + 16 }; })()`);
box.width += box.width % 2; box.height += box.height % 2;   // even size for H.264
fs.mkdirSync(outDir, { recursive: true });
let written = 0, after = 0;
for (let i = 0; i < +nFrames; i++) {
  await ev(`(() => { const b = document.getElementById("b-step"); for (let k = 0; k < ${+stepsPerFrame}; k++) b.click(); return 1; })()`);
  const shot = await send("Page.captureScreenshot", { format: "png", clip: { ...box, scale: 1 } });
  fs.writeFileSync(`${outDir}/f${String(i).padStart(4, "0")}.png`, Buffer.from(shot.result.data, "base64"));
  written = i + 1;
  // stop about one second after the game has ended
  const st = await ev(`document.getElementById("s-status").textContent`);
  if (!["paused", "playing", "flying", "ready"].includes(st)) { if (++after > 18) break; }
}
const status = await ev(`document.getElementById("s-status").textContent`);
console.log(JSON.stringify({ frames: written, box, status }));
ws.close();
