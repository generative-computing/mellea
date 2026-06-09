import {readFileSync, writeFileSync} from 'node:fs';

const [, , version] = process.argv;
if (!version) {
  console.error('Usage: set-last-version.mjs <version>');
  process.exit(1);
}

const configPath = 'docs/docusaurus.config.ts';
let src = readFileSync(configPath, 'utf8');

// Update lastVersion to the new release tag
const updated = src.replace(
  /lastVersion:\s*'[^']*',/,
  `lastVersion: '${version}',`,
);
if (updated === src) {
  console.error(`error: lastVersion line not found in ${configPath}`);
  process.exit(1);
}
src = updated;

// Add path: 'main' to the current version block if not already present.
// This is a one-time insertion needed once a snapshot version exists as the
// default — before the first snapshot, current IS the default and needs no path.
if (!src.includes("path: 'main'")) {
  src = src.replace(
    /label: 'main \(unreleased\)',/,
    "label: 'main (unreleased)',\n              path: 'main',",
  );
  console.log("Added path: 'main' to versions.current");
}

writeFileSync(configPath, src);
console.log(`Set lastVersion to '${version}' in ${configPath}`);
