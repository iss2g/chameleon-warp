import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import pkg from './package.json'

export default defineConfig({
  plugins: [react()],
  // Inject the app version at build time so the frontend can show a one-time
  // changelog popup when it changes (compared against localStorage).
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  build: {
    // three.js alone is ~600 kB minified; the app is served locally, so one
    // bundle is fine and the default 500 kB warning is just noise.
    chunkSizeWarningLimit: 1500,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
