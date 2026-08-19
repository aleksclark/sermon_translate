import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

const devHost = process.env.VITE_DEV_HOST || undefined
const devPort = Number(process.env.VITE_DEV_PORT ?? '5173')
const strictPort = process.env.VITE_STRICT_PORT === '1'

const hmrHost = process.env.HMR_HOST?.trim() || undefined
const hmrClientPort = process.env.HMR_CLIENT_PORT
  ? Number(process.env.HMR_CLIENT_PORT)
  : undefined
const hmrProtocol = process.env.HMR_PROTOCOL?.trim() || undefined

const hmr =
  hmrHost != null
    ? {
        host: hmrHost,
        clientPort: hmrClientPort ?? 80,
        protocol: (hmrProtocol ?? 'ws') as 'ws' | 'wss',
        port: Number.isFinite(devPort) ? devPort : 5173,
      }
    : undefined

// Compose sets DEV_ALLOWED_HOSTS / STACKLANE_BASE_DOMAIN; host `pnpm dev` keeps Vite defaults.
const configuredBase = (process.env.STACKLANE_BASE_DOMAIN ?? 'test').trim() || 'test'
const extraHosts = (process.env.DEV_ALLOWED_HOSTS ?? '')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean)
const allowCustomHosts =
  process.env.DEV_ALLOWED_HOSTS != null || process.env.STACKLANE_BASE_DOMAIN != null
const allowedHosts = allowCustomHosts
  ? Array.from(
      new Set([`.${configuredBase}`, '.test', 'localhost', '127.0.0.1', ...extraHosts]),
    )
  : undefined

export default defineConfig({
  plugins: [react()],
  server: {
    ...(devHost ? { host: devHost } : {}),
    port: Number.isFinite(devPort) ? devPort : 5173,
    ...(strictPort ? { strictPort: true } : {}),
    ...(allowedHosts ? { allowedHosts } : {}),
    ...(hmr ? { hmr } : {}),
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/ws': {
        target: apiProxyTarget.replace(/^http/i, 'ws'),
        ws: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
