#!/usr/bin/env node

/**
 * Reset a user's password in the database.
 *
 * Usage:
 *   node scripts/reset-password.mjs <email> <new-password>
 *
 * Example:
 *   node scripts/reset-password.mjs user@example.com myNewPassword123
 *
 * Requires DATABASE_URL to be set (reads from .env automatically).
 */

import { hashPassword } from "better-auth/crypto";
import pg from "pg";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from dashboard root
function loadEnv() {
  try {
    const envPath = resolve(__dirname, "../.env");
    const content = readFileSync(envPath, "utf-8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIndex = trimmed.indexOf("=");
      if (eqIndex === -1) continue;
      const key = trimmed.slice(0, eqIndex).trim();
      const val = trimmed.slice(eqIndex + 1).trim().replace(/^["']|["']$/g, "");
      if (!process.env[key]) process.env[key] = val;
    }
  } catch {
    // .env not found, rely on existing env vars
  }
}

loadEnv();

const [email, newPassword] = process.argv.slice(2);

if (!email || !newPassword) {
  console.error("Usage: node scripts/reset-password.mjs <email> <new-password>");
  process.exit(1);
}

if (newPassword.length < 6) {
  console.error("Error: Password must be at least 6 characters.");
  process.exit(1);
}

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  console.error("Error: DATABASE_URL is not set. Set it in .env or as an environment variable.");
  process.exit(1);
}

const client = new pg.Client({ connectionString: databaseUrl });

try {
  await client.connect();

  // Find the user
  const userResult = await client.query(
    `SELECT id, email, name FROM "user" WHERE email = $1`,
    [email]
  );

  if (userResult.rows.length === 0) {
    console.error(`Error: No user found with email "${email}".`);
    process.exit(1);
  }

  const user = userResult.rows[0];
  console.log(`Found user: ${user.name || "(no name)"} <${user.email}>`);

  // Hash the new password
  const hash = await hashPassword(newPassword);

  // Update the password in the account table
  const updateResult = await client.query(
    `UPDATE account SET password = $1, updated_at = now()
     WHERE provider_id = 'credential' AND user_id = $2`,
    [hash, user.id]
  );

  if (updateResult.rowCount === 0) {
    console.error("Error: No credential account found for this user. They may have signed up with a different provider.");
    process.exit(1);
  }

  console.log(`Password updated successfully for ${user.email}.`);
} catch (err) {
  console.error("Failed:", err.message);
  process.exit(1);
} finally {
  await client.end();
}
