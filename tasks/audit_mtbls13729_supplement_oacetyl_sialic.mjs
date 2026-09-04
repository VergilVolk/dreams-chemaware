import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const sourceDir = path.join(root, "data", "mtbls13729", "source_paper_supplements");
const workbookNames = [2, 3, 4, 5, 6].map(
  (n) => `pr5c01260_si_00${n}.xlsx`,
);
const csvNames = [7, 8, 9].map((n) => `pr5c01260_si_00${n}.csv`);

const textPattern = /(o[\s-]*acetyl|acneu|neu5[,\s-]*9|acetylneuramin|sialic|neu5ac)/i;
const targets = [
  { label: "mono-O-acetyl-Neu5Ac_[M-H]-", mz: 350.1093 },
  { label: "di-O-acetyl-Neu5Ac_[M-H]-", mz: 392.1199 },
  { label: "tri-O-acetyl-Neu5Ac_[M-H]-", mz: 434.1305 },
];
const ppmTolerance = 10;

function textHit(value) {
  return typeof value === "string" && textPattern.test(value);
}

function numericHits(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return [];
  return targets.filter((target) =>
    Math.abs(value - target.mz) <= target.mz * ppmTolerance * 1e-6,
  );
}

function scanRows(rows, source, sheet) {
  const hits = [];
  for (let r = 0; r < rows.length; r += 1) {
    const row = rows[r] ?? [];
    const rowText = row.map((v) => (v == null ? "" : String(v))).join(" | ");
    for (let c = 0; c < row.length; c += 1) {
      const value = row[c];
      if (textHit(value)) {
        hits.push({ source, sheet, row: r + 1, column: c + 1, kind: "text", value, row_context: rowText });
      }
      for (const target of numericHits(value)) {
        hits.push({
          source,
          sheet,
          row: r + 1,
          column: c + 1,
          kind: "numeric_mz",
          value,
          target: target.label,
          ppm_error: ((value - target.mz) / target.mz) * 1e6,
          row_context: rowText,
        });
      }
    }
  }
  return hits;
}

const report = { status: "mtbls13729_supplement_oacetyl_sialic_audit_complete", ppm_tolerance: ppmTolerance, files: [], hits: [] };

for (const name of workbookNames) {
  const filePath = path.join(sourceDir, name);
  const blob = await FileBlob.load(filePath);
  const workbook = await SpreadsheetFile.importXlsx(blob);
  const sheets = [];
  for (const worksheet of workbook.worksheets.items) {
    const used = worksheet.getUsedRange(true);
    const values = used ? used.values : [];
    sheets.push({ name: worksheet.name, rows: values.length, columns: values.reduce((m, row) => Math.max(m, row?.length ?? 0), 0) });
    report.hits.push(...scanRows(values, name, worksheet.name));
  }
  report.files.push({ name, sheets });
}

for (const name of csvNames) {
  const filePath = path.join(sourceDir, name);
  const text = await fs.readFile(filePath, "utf8");
  const lines = text.split(/\r?\n/);
  const relevant = lines
    .map((line, index) => ({ line, index: index + 1 }))
    .filter(({ line }) => textPattern.test(line));
  report.files.push({ name, rows: lines.length });
  for (const hit of relevant) {
    report.hits.push({ source: name, sheet: null, row: hit.index, column: null, kind: "text", value: hit.line, row_context: hit.line });
  }
}

report.summary = {
  total_hits: report.hits.length,
  text_hits: report.hits.filter((hit) => hit.kind === "text").length,
  numeric_mz_hits: report.hits.filter((hit) => hit.kind === "numeric_mz").length,
  files_with_hits: [...new Set(report.hits.map((hit) => hit.source))],
};

const outputPath = path.join(root, "data", "mtbls13729", "source_paper_supplements", "oacetyl_sialic_audit_v1.json");
await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report.summary, null, 2));
console.log(`Saved: ${outputPath}`);
