#!/usr/bin/env node
// OmniAI's JSON accuracy over the recorded replies, the page's primary metric.
//
//   npm ci && node score.mjs              # the recorded run, from evidence/
//   node score.mjs runs                   # your own run.py output as well
//
// calculateJsonAccuracy, countChanges and countTotalFields are OmniAI's code from
// github.com/getomni-ai/benchmark src/evaluation/json.ts (MIT), ported from
// TypeScript by removing the type annotations and nothing else, and run with the
// json-diff version its package.json pins (^1.0.6, locked here at 1.0.6).
// ignoreCases is false, as Omni's runner calls it. A reply that is not JSON, or
// an error, scores 0.
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import jsonDiff from 'json-diff';

const { diff } = jsonDiff;

// ---- OmniAI src/evaluation/json.ts, verbatim logic -------------------------
const calculateJsonAccuracy = (actual, predicted, ignoreCases = false) => {
	const processedActual = ignoreCases ? convertStringsToUppercase(actual) : actual;
	const processedPredicted = ignoreCases ? convertStringsToUppercase(predicted) : predicted;
	const fullDiffResult = diff(processedActual, processedPredicted, { full: true, sort: true });
	const diffResult = diff(processedActual, processedPredicted, { sort: true });
	const totalFields = countTotalFields(processedActual);
	if (!diffResult) {
		return { score: 1, jsonDiff: {}, fullJsonDiff: {}, jsonDiffStats: { additions: 0, deletions: 0, modifications: 0, total: 0 }, totalFields };
	}
	const changes = countChanges(diffResult);
	const score = Math.max(0, 1 - (changes.additions + changes.deletions + changes.modifications) / totalFields);
	return { score: Number(score.toFixed(4)), jsonDiff: diffResult, fullJsonDiff: fullDiffResult, jsonDiffStats: changes, totalFields };
};
const convertStringsToUppercase = (obj) => {
	if (obj === null || typeof obj !== 'object') return obj;
	if (Array.isArray(obj)) return obj.map((item) => convertStringsToUppercase(item));
	const result = {};
	for (const key in obj) {
		const value = obj[key];
		if (typeof value === 'string') result[key] = value.toUpperCase();
		else if (typeof value === 'object' && value !== null) result[key] = convertStringsToUppercase(value);
		else result[key] = value;
	}
	return result;
};
const countChanges = (diffResult) => {
	const changes = { additions: 0, deletions: 0, modifications: 0, total: 0 };
	const traverse = (obj) => {
		if (!obj || typeof obj !== 'object') return;
		for (const key in obj) {
			const value = obj[key];
			if (Array.isArray(value)) {
				value.forEach((item) => {
					if (!Array.isArray(item) || item.length !== 2) return;
					const [operation, element] = item;
					if (element === null || typeof element !== 'object') {
						switch (operation) {
							case '+': changes.additions++; break;
							case '-': changes.deletions++; break;
						}
					} else {
						switch (operation) {
							case '+': changes.additions += countTotalFields(element); break;
							case '-': changes.deletions += countTotalFields(element); break;
							case '~': traverse(element); break;
						}
					}
				});
			} else {
				if (key.endsWith('__deleted')) {
					if (value === null || typeof value !== 'object') changes.deletions++;
					else changes.deletions += countTotalFields(value);
				} else if (key.endsWith('__added')) {
					if (value === null || typeof value !== 'object') changes.additions++;
					else changes.additions += countTotalFields(value);
				} else if (typeof value === 'object' && value !== null) {
					if (value.__old !== undefined && value.__new !== undefined) {
						if (value.__old === null && value.__new !== null) changes.modifications += countTotalFields(value.__new) || 1;
						else changes.modifications += countTotalFields(value.__old) || 1;
					} else {
						traverse(value);
					}
				}
			}
		}
	};
	traverse(diffResult);
	changes.total = changes.additions + changes.deletions + changes.modifications;
	return changes;
};
function countTotalFields(obj) {
	let count = 0;
	const traverse = (current) => {
		if (!current || typeof current !== 'object') return;
		if (Array.isArray(current)) {
			current.forEach((item) => {
				if (typeof item === 'object' && item !== null) traverse(item);
				else count++;
			});
		} else {
			for (const key in current) {
				if (key.includes('__')) continue;
				if (current[key] === null || typeof current[key] === 'string' || typeof current[key] === 'number' || typeof current[key] === 'boolean') count++;
				else if (typeof current[key] === 'object') traverse(current[key]);
			}
		}
	};
	traverse(obj);
	return count;
}
// ---- end of OmniAI's code ---------------------------------------------------

// One step before Omni's code, applied to the gold and to every arm alike: a key
// the schema declares but the JSON leaves out is filled in as null. Strict modes
// (OpenAI, SIE) must return every key and say null; Anthropic's form leaves the
// key out; some gold records omit a key and some write null. Without this, the
// same answer would score differently by vendor. Nothing else is touched.
const typeOf = (s) => {
	let t = s?.type;
	if (Array.isArray(t)) t = t.find((x) => x !== 'null');
	return t;
};
function fillNulls(value, schema) {
	if (!schema || value === null || value === undefined) return value;
	const t = typeOf(schema);
	if (t === 'object' && typeof value === 'object' && !Array.isArray(value)) {
		const out = { ...value };
		for (const [k, sub] of Object.entries(schema.properties ?? {})) {
			out[k] = k in out ? fillNulls(out[k], sub) : null;
		}
		return out;
	}
	if (t === 'array' && Array.isArray(value) && schema.items) return value.map((v) => fillNulls(v, schema.items));
	return value;
}

const sets = JSON.parse(readFileSync('evidence/inputs/sets.json', 'utf8'));
const schemas = new Map(sets.documents.map((d) => [d.id, d.schema]));
const gold = new Map(sets.documents.map((d) => [d.id, fillNulls(d.gold, d.schema)]));
const omniIds = sets.documents.filter((d) => d.set === 'omni').map((d) => d.id);

function load(path) {
	const rows = new Map();
	for (const line of readFileSync(path, 'utf8').split('\n')) {
		if (!line.trim()) continue;
		const row = JSON.parse(line);
		const prev = rows.get(row.id);
		if (!prev || prev.error) rows.set(row.id, row);
	}
	return rows;
}

function score(rows, ids) {
	let total = 0;
	for (const id of ids) {
		const row = rows.get(id);
		let parsed = null;
		if (row && !row.error) {
			try { parsed = JSON.parse(row.text); } catch { parsed = null; }
		}
		if (parsed !== null && typeof parsed === 'object') total += calculateJsonAccuracy(gold.get(id), fillNulls(parsed, schemas.get(id))).score;
	}
	return total / ids.length;
}

const dirs = [join('evidence', 'runs', 'e1'), join('evidence', 'runs', 'e1-claude-optional-form'), ...process.argv.slice(2)];
console.log(`OmniAI JSON accuracy on ${omniIds.length} business documents`);
for (const dir of dirs) {
	if (!existsSync(dir)) continue;
	for (const file of readdirSync(dir).filter((f) => f.endsWith('.jsonl')).sort()) {
		const rows = load(join(dir, file));
		const ids = dir.startsWith('evidence') ? omniIds : omniIds.filter((id) => rows.has(id));
		if (!ids.length) continue;
		console.log(`  ${(100 * score(rows, ids)).toFixed(1).padStart(5)}%  ${file.replace('.jsonl', '')}  (${dir}, ${ids.length} documents)`);
	}
}
