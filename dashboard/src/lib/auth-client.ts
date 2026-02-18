/**
 * Better Auth — client-side instance.
 *
 * Used in React components for signIn, signUp, useSession, signOut.
 * Talks to the dashboard's own /api/auth/* endpoints (same origin, no baseURL needed).
 */

import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient();

export const { signIn, signUp, signOut, useSession } = authClient;
