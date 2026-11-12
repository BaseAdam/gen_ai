import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// In development the API runs on its own (uv run fastapi dev); Vite forwards API calls to it.
const API_URL = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/agent': API_URL },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
