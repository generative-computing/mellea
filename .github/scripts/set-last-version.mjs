// Copyright IBM Corp. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

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

writeFileSync(configPath, src);
console.log(`Set lastVersion to '${version}' in ${configPath}`);
