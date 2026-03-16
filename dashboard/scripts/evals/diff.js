const fs = require("node:fs");
const path = require("node:path");

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      out[key] = true;
      continue;
    }
    out[key] = next;
    i += 1;
  }
  return out;
}

function loadJSON(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function metric(obj, key) {
  return Number(obj?.summary?.[key] || 0);
}

function main() {
  const args = parseArgs(process.argv);
  const baselinePath = path.resolve(process.cwd(), args.baseline || "../.ci/evals/baseline.json");
  const currentPath = path.resolve(process.cwd(), args.current || "../.tmp/evals/latest.json");
  const outputPath = path.resolve(process.cwd(), args.output || "../.tmp/evals/diff.json");

  const current = loadJSON(currentPath);
  const baselineExists = fs.existsSync(baselinePath);
  const baseline = baselineExists ? loadJSON(baselinePath) : { summary: {} };

  const diff = {
    generated_at: new Date().toISOString(),
    baseline_found: baselineExists,
    metrics: {
      success_rate_delta: Number((metric(current, "success_rate") - metric(baseline, "success_rate")).toFixed(4)),
      missing_required_tool_rate_delta: Number((metric(current, "missing_required_tool_rate") - metric(baseline, "missing_required_tool_rate")).toFixed(4)),
      out_of_bounds_rate_delta: Number((metric(current, "out_of_bounds_rate") - metric(baseline, "out_of_bounds_rate")).toFixed(4)),
      recovery_rate_delta: Number((metric(current, "recovery_rate") - metric(baseline, "recovery_rate")).toFixed(4)),
    },
  };

  ensureDir(outputPath);
  fs.writeFileSync(outputPath, JSON.stringify(diff, null, 2) + "\n", "utf8");
  process.stdout.write(`Eval diff complete. Baseline found: ${baselineExists}\n`);
}

main();
