import {readFileSync, writeFileSync} from 'node:fs';

const [, , version] = process.argv;
if (!version) {
  console.error('Usage: set-last-version.mjs <version>');
  process.exit(1);
}

const path = 'docs/docusaurus.config.ts';
const src = readFileSync(path, 'utf8');
const next = src.replace(
  /lastVersion:\s*'[^']*',/,
  `lastVersion: '${version}',`,
);
if (next === src) {
  console.error(`error: lastVersion line not found in ${path}`);
  process.exit(1);
}
writeFileSync(path, next);
console.log(`Set lastVersion to '${version}' in ${path}`);
