if (typeof window !== "undefined") {
  throw new Error("lib/auth.ts is server-only — do not import from client code");
}

import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "#/generated/prisma/client.js";

const isDev = process.env.NODE_ENV !== "production";

const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error("DATABASE_URL is required");
}

const adapter = new PrismaPg({ connectionString });
const prisma = new PrismaClient({ adapter });

const authSecret =
  process.env.BETTER_AUTH_SECRET || process.env.AUTH_SECRET;
if (!authSecret && !isDev) {
  throw new Error("BETTER_AUTH_SECRET is required in production");
}

const authBaseURL =
  process.env.BETTER_AUTH_BASE_URL ||
  process.env.BETTER_AUTH_URL ||
  "http://localhost:3000";

export const auth = betterAuth({
  secret: authSecret || "aether-dev-secret-local",
  baseURL: authBaseURL,

  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),

  emailAndPassword: {
    enabled: true,
    minPasswordLength: 6,
  },

  session: {
    expiresIn: 60 * 60 * 24 * 30, // 30 days
    updateAge: 60 * 60 * 24, // refresh session token daily
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5, // 5 min cookie cache
    },
  },

  trustedOrigins: [
    "http://localhost:3000",
    "http://localhost:4000",
    authBaseURL,
    ...(process.env.BETTER_AUTH_TRUSTED_ORIGINS?.split(",").map((s) => s.trim()) || []),
  ].filter(Boolean),
});

export type Session = typeof auth.$Infer.Session;
