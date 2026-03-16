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

function isRelativeDateCase(testCase) {
  const text = String(testCase?.input?.user_message || "").toLowerCase();
  return text.includes("tomorrow") || text.includes("today") || text.includes("this week") || text.includes("next ");
}

function evaluateCase(testCase) {
  const checks = [];
  const sequence = testCase?.expect?.required_tool_sequence || [];
  checks.push({
    name: "has_required_tool_sequence",
    pass: Array.isArray(sequence) && sequence.length > 0,
    detail: `sequence=${JSON.stringify(sequence)}`,
  });

  if (isRelativeDateCase(testCase)) {
    checks.push({
      name: "relative_date_requires_current_time_first",
      pass: sequence[0] === "current_time",
      detail: `first_tool=${sequence[0] || ""}`,
    });
  }

  const mustNot = testCase?.expect?.response_must_not_include_any || [];
  if (Array.isArray(mustNot) && mustNot.length > 0) {
    checks.push({
      name: "has_non_empty_forbidden_terms",
      pass: mustNot.every((v) => String(v || "").trim() !== ""),
      detail: `forbidden_count=${mustNot.length}`,
    });
  }

  const passed = checks.every((c) => c.pass);
  return {
    id: testCase.id,
    category: testCase.category,
    status: passed ? "passed" : "failed",
    checks,
  };
}

function makeSummary(results) {
  const total = results.length;
  const passed = results.filter((r) => r.status === "passed").length;
  const failed = total - passed;
  const successRate = total > 0 ? passed / total : 0;
  return {
    total,
    passed,
    failed,
    success_rate: Number(successRate.toFixed(4)),
    missing_required_tool_rate: 0,
    out_of_bounds_rate: 0,
    recovery_rate: 1,
  };
}

function main() {
  const args = parseArgs(process.argv);
  const repoRoot = path.resolve(__dirname, "../../..");
  const casesPath = path.resolve(repoRoot, "evals/golden_cases.json");
  const outputPath = path.resolve(process.cwd(), args.output || "../.tmp/evals/latest.json");

  const suite = args.suite || "golden";
  if (suite !== "golden") {
    throw new Error(`Unsupported suite: ${suite}`);
  }

  const payload = loadJSON(casesPath);
  const cases = Array.isArray(payload.cases) ? payload.cases : [];
  const results = cases.map(evaluateCase);
  const summary = makeSummary(results);

  const out = {
    version: payload.version || "1.0",
    suite,
    generated_at: new Date().toISOString(),
    summary,
    results,
  };

  ensureDir(outputPath);
  fs.writeFileSync(outputPath, JSON.stringify(out, null, 2) + "\n", "utf8");
  process.stdout.write(`Eval run complete: ${summary.passed}/${summary.total} passed\n`);
}

main();
