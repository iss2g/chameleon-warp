/**
 * User-facing changelog. Shown in the Changelog info page and, once per version
 * bump, in an auto-opening popup (compared against localStorage).
 *
 * Keep entries newest-first. `version` should match package.json when you cut a
 * release — the popup fires when the built version differs from what the
 * browser last saw.
 */
export interface ChangeEntry {
  version: string
  date: string // ISO date
  items: string[]
}

export const CHANGELOG: ChangeEntry[] = [
  {
    version: '0.5.0',
    date: '2026-10-07',
    items: [
      'First public release. Runs on your own machine — your meshes never leave it.',
      'Five tools, each a step-by-step wizard with an in-app guide: Wrap, Wrap Region, DT Transfer, Refit (beta) and Fit (beta).',
      'Interactive marker picking with symmetric mode, undo/redo and JSON import/export; no markers needed when source and target share topology.',
      'Results keep the original quads and UVs. Residual heatmap with mean / p95 / max distance after every wrap.',
      'FBX in and out (with Blender): shape keys become poses on upload, DT results download as one FBX with shape keys.',
      'Refit: placement gizmo, Poke pins, grip preview, freeze paint, automatic layer detection, body mask editing and proxy wardrobe.',
      'Long jobs run in the background with progress — close the tab and come back.',
      'One-command setup: Docker Compose, or setup/start scripts for Linux, macOS and Windows.',
    ],
  },
]

/** The current build version, injected by Vite (see vite.config.ts). */
export const APP_VERSION: string =
  typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '0.0.0'
