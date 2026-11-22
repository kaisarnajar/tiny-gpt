import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During `npm run dev`, send /api calls to the Python server on port 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
