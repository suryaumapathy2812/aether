"use client";

import { useEffect } from "react";
import { useSession } from "@/lib/auth-client";
import { setSessionToken } from "@/lib/api";

/**
 * Syncs the better-auth session token into the API client module.
 *
 * better-auth manages auth via httpOnly cookies (dashboard ↔ /api/auth/*).
 * But the orchestrator is on a different origin, so we can't use cookies.
 * Instead, we read the session token from useSession() and pass it as
 * an Authorization: Bearer header to the orchestrator.
 *
 * This component should be rendered in the root layout.
 */
export default function SessionSync() {
  const { data: session } = useSession();

  useEffect(() => {
    if (session?.session?.token) {
      setSessionToken(session.session.token);
    } else {
      setSessionToken(null);
    }
  }, [session]);

  return null;
}
