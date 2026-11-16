// Helper for tests/test_engine_parity.py: runs app/static/engine.js under Node on the cases in
// the JSON file given as argv[2] and prints the results as JSON.
const fs = require("fs");
const path = require("path");
const E = require(path.join(__dirname, "..", "app", "static", "engine.js"));

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = { inputs: [], segments: [], grids: [] };

for (const c of cases.clips) {
  const rows = E.modelInput(c.x, c.face, cases.cfg);
  out.inputs.push(rows.map((r) => Array.from(r)));
}

for (const s of cases.streams) {
  const seg = new E.Segmenter();
  const events = [];
  s.x.forEach((f, i) => {
    const done = seg.push(f, s.t[i], s.face[i]);
    if (done) events.push([i, done.length, seg.endReason]);
  });
  const tail = seg.flush();
  if (tail) events.push([s.x.length, tail.length, seg.endReason]);
  out.segments.push(events);
}

for (const n of cases.grid_sizes) {
  out.grids.push([E.linspaceInts(0, Math.floor(n * 0.45), 10),
                  E.linspaceInts(Math.floor(n * 0.65), n, 7)]);
}

process.stdout.write(JSON.stringify(out));
