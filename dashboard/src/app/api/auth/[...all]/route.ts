/**
 * Better Auth catch-all route handler.
 * Handles all /api/auth/* requests (signin, signup, session, etc.)
 */

import { auth } from "@/lib/auth";
import { toNextJsHandler } from "better-auth/next-js";

export const { POST, GET } = toNextJsHandler(auth);
