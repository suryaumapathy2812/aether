"use client";

import { useEffect } from "react";

/**
 * Registers the service worker on mount.
 *
 * next-pwa v5 is unreliable with Next.js App Router — it sometimes
 * fails to inject the registration script. This component ensures
 * /sw.js is always registered in production.
 */
export default function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      "serviceWorker" in navigator &&
      process.env.NODE_ENV === "production"
    ) {
      navigator.serviceWorker.register("/sw.js").catch((err) => {
        console.warn("SW registration failed:", err);
      });
    }
  }, []);

  return null;
}
