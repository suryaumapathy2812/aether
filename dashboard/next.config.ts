import type { NextConfig } from "next";
// next-pwa ships no TypeScript declarations — use require to bypass strict check
// eslint-disable-next-line @typescript-eslint/no-require-imports
const withPWA = require("next-pwa");

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: [
    "dashboard.core-ai.orb.local",
  ],
};

export default withPWA({
  dest: "public",
  // Only activate service worker in production — avoids cache confusion in dev
  disable: process.env.NODE_ENV === "development",
  // Don't precache Next.js build manifests (they change every build)
  buildExcludes: [/app-build-manifest\.json$/, /middleware-manifest\.json$/],
  // Cache the background image and other static assets
  runtimeCaching: [
    {
      // App shell pages — network first, fall back to cache
      urlPattern: /^https?.*/,
      handler: "NetworkFirst",
      options: {
        cacheName: "aether-runtime",
        expiration: { maxEntries: 64, maxAgeSeconds: 24 * 60 * 60 },
        networkTimeoutSeconds: 10,
      },
    },
  ],
})(nextConfig);
