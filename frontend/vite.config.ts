import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // REST endpoints forwarded to FastAPI
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/training": { target: "http://localhost:8000", changeOrigin: true },
      "/metrics": { target: "http://localhost:8000", changeOrigin: true },
      "/drone-state": { target: "http://localhost:8000", changeOrigin: true },
      "/symbolic-log": { target: "http://localhost:8000", changeOrigin: true },
      // WebSocket upgrade
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
