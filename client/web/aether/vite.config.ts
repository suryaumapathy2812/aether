import { defineConfig } from "vite-plus";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { nitro } from "nitro/vite";

const ORCHESTRATOR = process.env.ORCHESTRATOR_BASE_URL || "http://localhost:4000";

const config = defineConfig({
  lint: { options: { typeAware: true, typeCheck: true } },
  plugins: [
    nitro({ rollupConfig: { external: [/^@sentry\//] } }),
    tailwindcss(),
    tanstackStart({ ssr: false }),
    viteReact({
      babel: {
        plugins: ["babel-plugin-react-compiler"],
      },
    }),
  ],
  server: {
    proxy: {
      "/api/go": {
        target: ORCHESTRATOR,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/go/, ""),
      },
      "/agent/v1": {
        target: ORCHESTRATOR,
        changeOrigin: true,
      },
      "/go/v1": {
        target: ORCHESTRATOR,
        changeOrigin: true,
      },
      "/ws": {
        target: ORCHESTRATOR.replace("http://", "ws://"),
        ws: true,
      },
    },
  },
});

export default config;
