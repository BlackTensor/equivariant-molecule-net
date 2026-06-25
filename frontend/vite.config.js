import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standard Vite + React config. The dev server runs on port 5173 (already
// allow-listed by the FastAPI CORS middleware in src/api/main.py), so the
// browser can talk to the backend at http://localhost:8000 during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
