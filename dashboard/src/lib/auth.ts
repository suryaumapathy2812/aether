/**
 * Better Auth — server-side instance.
 *
 * Runs inside the Next.js server. Handles signup, signin, session management.
 * Uses Prisma adapter pointing at the shared Postgres (same DB as orchestrator).
 *
 * The orchestrator validates sessions by reading the `session` table directly —
 * no need for a separate token system.
 */

import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "@/generated/prisma/client";

const adapter = new PrismaPg({
  connectionString: process.env.DATABASE_URL || "",
});

const prisma = new PrismaClient({ adapter });

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),

  emailAndPassword: {
    enabled: true,
    minPasswordLength: 6,
  },

  session: {
    expiresIn: 60 * 60 * 24 * 30, // 30 days
    updateAge: 60 * 60 * 24,       // refresh session token daily
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5 min cookie cache
    },
  },

  trustedOrigins: [
    "http://localhost:3000",
    "http://localhost:9000",
    process.env.BETTER_AUTH_URL || "",
    // Additional trusted origins (comma-separated) for OrbStack, tunnels, etc.
    ...(process.env.BETTER_AUTH_TRUSTED_ORIGINS?.split(",").map(s => s.trim()) || []),
  ].filter(Boolean),
});

export type Session = typeof auth.$Infer.Session;
