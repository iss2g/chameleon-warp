/**
 * Static info pages (About / Credits / What's new), rendered as a footer bar +
 * a modal overlay. Self-contained: manages its own open state, no router
 * dependency. Drop <InfoPages /> once at the bottom of App.
 */
import { useEffect, useState } from 'react'
import { CHANGELOG, APP_VERSION } from './changelog'

type PageKey = 'about' | 'license' | 'changelog'

const REPO_URL = 'https://github.com/iss2g/chameleon-warp'

const TABS: { key: PageKey; label: string }[] = [
  { key: 'about', label: 'About' },
  { key: 'license', label: 'Credits' },
  { key: 'changelog', label: "What's new" },
]

function AboutPage() {
  return (
    <>
      <h2>About Chameleon Warp</h2>
      <p>
        <b>Chameleon Warp</b> is an open-source tool for conforming one 3D mesh
        onto another, transferring pose and blendshape deformations between
        characters, and fitting garments onto a body — built on the
        Sumner–Popović deformation-transfer method.
      </p>
      <p>
        It runs entirely on your machine: the browser UI talks to a local
        backend, and your meshes never leave your computer.
      </p>
      <h3>How it works</h3>
      <ol>
        <li>Pick a tool: Wrap, Wrap Region, DT Transfer, Refit or Fit.</li>
        <li>Load a <b>source</b> mesh (left) and a <b>target</b> mesh (right).</li>
        <li>Click matching feature points on both to place <b>markers</b>.</li>
        <li>Run the tool, inspect the result, download OBJ / ZIP / FBX.</li>
      </ol>
      <p className="subtle">
        Heavy jobs run in the background with a progress bar, so you can close
        the tab and come back — the backend keeps computing. Version {APP_VERSION}.{' '}
        <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
          Source code &amp; docs
        </a>
        .
      </p>
    </>
  )
}

function LicensePage() {
  return (
    <>
      <h2>Credits</h2>
      <p>
        <b>Chameleon Warp</b> is released under the MIT License. It is built on
        open-source work, gratefully credited below.
      </p>
      <h3>Built on</h3>
      <p>
        The mesh algorithm core is derived from{' '}
        <a
          href="https://github.com/mickare/Deformation-Transfer-for-Triangle-Meshes"
          target="_blank"
          rel="noopener noreferrer"
        >
          Deformation-Transfer-for-Triangle-Meshes
        </a>{' '}
        by Michael Käser &amp; Jasmin Hoffmann, released under the MIT License
        (© 2021). It implements:
      </p>
      <p className="subtle">
        Robert W. Sumner and Jovan Popović, “Deformation Transfer for Triangle
        Meshes,” ACM Transactions on Graphics (SIGGRAPH), 2004.
      </p>
    </>
  )
}

function ChangelogPage() {
  return (
    <>
      <h2>What’s new</h2>
      {CHANGELOG.map((entry) => (
        <div key={entry.version} style={{ marginBottom: 16 }}>
          <h3>
            v{entry.version} <span className="subtle">· {entry.date}</span>
          </h3>
          <ul>
            {entry.items.map((it, i) => (
              <li key={i}>{it}</li>
            ))}
          </ul>
        </div>
      ))}
    </>
  )
}

const PAGES: Record<PageKey, () => JSX.Element> = {
  about: AboutPage,
  license: LicensePage,
  changelog: ChangelogPage,
}

export default function InfoPages({ openOnMount }: { openOnMount?: PageKey | null }) {
  const [open, setOpen] = useState<PageKey | null>(null)

  // openOnMount is computed by the parent in an effect (after our first
  // render), so react to it changing rather than only reading it once.
  useEffect(() => {
    if (openOnMount) setOpen(openOnMount)
  }, [openOnMount])

  // Close on Escape.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const Body = open ? PAGES[open] : null

  return (
    <>
      <footer className="info-footer">
        {TABS.map((t, i) => (
          <span key={t.key}>
            {i > 0 && <span className="info-footer-sep"> · </span>}
            <button className="link" onClick={() => setOpen(t.key)}>
              {t.label}
            </button>
          </span>
        ))}
      </footer>

      {open && Body && (
        <div className="info-overlay" onClick={() => setOpen(null)}>
          <div className="info-card" onClick={(e) => e.stopPropagation()}>
            <div className="info-tabs">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  className={t.key === open ? 'info-tab active' : 'info-tab'}
                  onClick={() => setOpen(t.key)}
                >
                  {t.label}
                </button>
              ))}
              <button className="info-close" onClick={() => setOpen(null)} aria-label="Close">
                ✕
              </button>
            </div>
            <div className="info-content">
              <Body />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
