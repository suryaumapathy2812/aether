import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: [
    "dashboard.core-ai.orb.local",
  ],
};

export default nextConfig;
