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

function asNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function main() {
  const args = parseArgs(process.argv);
  const inputPath = path.resolve(process.cwd(), args.input || "../.tmp/evals/latest.json");
  const report = loadJSON(inputPath);
  const summary = report.summary || {};

  const minSuccessRate = asNumber(args["min-success-rate"], 0.95);
  const maxMissingRequiredToolRate = asNumber(args["max-missing-required-tool-rate"], 0.02);
  const maxOutOfBoundsRate = asNumber(args["max-out-of-bounds-rate"], 0.0);
  const minRecoveryRate = asNumber(args["min-recovery-rate"], 0.9);

  const successRate = asNumber(summary.success_rate, 0);
  const missingRequiredToolRate = asNumber(summary.missing_required_tool_rate, 1);
  const outOfBoundsRate = asNumber(summary.out_of_bounds_rate, 1);
  const recoveryRate = asNumber(summary.recovery_rate, 0);

  if (successRate < minSuccessRate) {
    fail(`Gate failed: success_rate ${successRate} < ${minSuccessRate}`);
  }
  if (missingRequiredToolRate > maxMissingRequiredToolRate) {
    fail(`Gate failed: missing_required_tool_rate ${missingRequiredToolRate} > ${maxMissingRequiredToolRate}`);
  }
  if (outOfBoundsRate > maxOutOfBoundsRate) {
    fail(`Gate failed: out_of_bounds_rate ${outOfBoundsRate} > ${maxOutOfBoundsRate}`);
  }
  if (recoveryRate < minRecoveryRate) {
    fail(`Gate failed: recovery_rate ${recoveryRate} < ${minRecoveryRate}`);
  }

  process.stdout.write("Eval gate passed\n");
}

main();
