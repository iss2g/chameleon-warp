import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { sortPoses, useStore } from '../store'
import MeshViewport from './MeshViewport'

/**
 * Preview/animation layout.
 *
 * Two viewports side by side: the source pose (input we transferred from) and
 * the deformed target (the output). A timeline scrubber + play button below
 * binds them to the same frame index, so scrubbing animates both in lock-step.
 */
export default function PreviewPanel() {
  const session = useStore((s) => s.session)
  const tool = useStore((s) => s.tool)
  const lastRun = useStore((s) => s.lastRun)
  const poseMeshes = useStore((s) => s.poseMeshes)
  const resultMeshes = useStore((s) => s.resultMeshes)
  const sourceMesh = useStore((s) => s.sourceMesh)
  const targetMesh = useStore((s) => s.targetMesh)
  const wrapMesh = useStore((s) => s.wrapMesh)
  const resultFromRefit = useStore((s) => s.resultFromRefit)
  const bodyMask = useStore((s) => s.bodyMask)
  const refitStats = useStore((s) => s.refitStats)
  const showHeatmap = useStore((s) => s.showHeatmap)
  const setShowHeatmap = useStore((s) => s.setShowHeatmap)
  const previewLoading = useStore((s) => s.previewLoading)
  const previewError = useStore((s) => s.previewError)
  const selectedPoseIndex = useStore((s) => s.selectedPoseIndex)
  const isPlaying = useStore((s) => s.isPlaying)
  const fps = useStore((s) => s.fps)
  const showWireframe = useStore((s) => s.showWireframe)
  const setShowWireframe = useStore((s) => s.setShowWireframe)

  const saveBodyMask = useStore((s) => s.saveBodyMask)
  const dilateBodyMask = useStore((s) => s.dilateBodyMask)

  const setSelectedPoseIndex = useStore((s) => s.setSelectedPoseIndex)
  const togglePlay = useStore((s) => s.togglePlay)
  const setFps = useStore((s) => s.setFps)
  const loadPreviewMeshes = useStore((s) => s.loadPreviewMeshes)

  // Which per-vertex heatmap to draw: distance-to-body error, or (refit only)
  // the grip field the auto-detection built — so the user can see what the
  // fit decided to pull and what it left free.
  const [heatMode, setHeatMode] = useState<'error' | 'grip'>('error')

  // Combined preview: show the fitted garment ON the body. Defaults ON for
  // refit results (which ship a body), OFF otherwise. "Apply body mask" hides
  // the body faces covered by the garment.
  const [showBody, setShowBody] = useState(resultFromRefit)
  const [applyBodyMask, setApplyBodyMask] = useState(false)
  // Follow the result kind: a fresh refit turns the body on, a plain wrap off.
  useEffect(() => {
    setShowBody(resultFromRefit)
  }, [resultFromRefit])
  const bodyOn = showBody && resultFromRefit && !!targetMesh
  const maskOn = bodyOn && applyBodyMask && !!bodyMask

  // ---- body-mask paint editor (item 2) ----
  // editingMask swaps the right viewport to paint on the BODY. maskWorking is
  // the live edit set; computedBaseline is what "Reset" reverts to (the mask as
  // it stood when editing began). maskBusy guards async save/dilate.
  const [editingMask, setEditingMask] = useState(false)
  const [maskWorking, setMaskWorking] = useState<number[]>([])
  const [computedBaseline, setComputedBaseline] = useState<number[]>([])
  const [maskBusy, setMaskBusy] = useState(false)

  // Leaving a refit result (or losing the body) exits the editor.
  useEffect(() => {
    if (editingMask && !(resultFromRefit && targetMesh && bodyMask)) {
      setEditingMask(false)
    }
  }, [editingMask, resultFromRefit, targetMesh, bodyMask])

  const enterMaskEdit = () => {
    const base = bodyMask?.hidden_vertices ?? []
    setMaskWorking(base)
    setComputedBaseline(base)
    setShowHeatmap(false)   // heatmap would fight the mask highlight
    setEditingMask(true)
  }
  const onMaskPaint = (verts: number[], erase: boolean) => {
    setMaskWorking((prev) => {
      const s = new Set(prev)
      if (erase) for (const v of verts) s.delete(v)
      else for (const v of verts) s.add(v)
      return [...s]
    })
  }
  const doSaveMask = async () => {
    setMaskBusy(true)
    try {
      const m = await saveBodyMask(maskWorking)
      if (m) { setMaskWorking(m.hidden_vertices); setComputedBaseline(m.hidden_vertices) }
    } finally { setMaskBusy(false) }
  }
  const doDilate = async (rings: number) => {
    setMaskBusy(true)
    try {
      const m = await dilateBodyMask(rings, maskWorking)
      if (m) setMaskWorking(m.hidden_vertices)
    } finally { setMaskBusy(false) }
  }
  const bodyVertCount = bodyMask?.body_vertex_count ?? targetMesh?.vertex_count ?? 0

  // Client-side sanity warnings derived from the refit stats (item 3b).
  const refitWarnings = useMemo(() => {
    if (!refitStats || !wrapMesh) return [] as string[]
    const n = wrapMesh.vertex_count
    const w: string[] = []
    const num = (k: string) => (typeof refitStats[k] === 'number' ? (refitStats[k] as number) : undefined)
    const matched = num('matched_verts'), opposed = num('opposed_verts'), contact = num('contact_verts')
    if (matched === 0)
      w.push('Nothing gripped — check placement/scale or lower the contact thresholds.')
    if (opposed !== undefined && opposed > 0.5 * n)
      w.push('Most normals face away from the body — the garment (or body) may have inverted normals.')
    if (contact !== undefined && contact < 0.02 * n)
      w.push('Garment placed far from the body — move/scale it closer or raise contact_free.')
    return w
  }, [refitStats, wrapMesh])

  // ---- pose list filtered to those with computed results ----
  const sortedPoses = useMemo(
    () => (session ? sortPoses(session.poses) : []),
    [session],
  )
  const playablePoses = useMemo(
    () => sortedPoses.filter((p) => session?.result_pose_ids.includes(p.pose_id)),
    [sortedPoses, session],
  )
  const currentPoseId = playablePoses[selectedPoseIndex]?.pose_id

  // Refresh cache whenever the set of completed poses changes (e.g. after a
  // Run) — with the CURRENT frame prioritized, so it renders after one
  // pose+result pair instead of after the whole set. Frame changes re-enter
  // to re-prioritize; already-fetched/in-flight meshes are shared, not
  // re-downloaded.
  useEffect(() => {
    loadPreviewMeshes(currentPoseId)
  }, [lastRun, currentPoseId, loadPreviewMeshes])

  // Clamp selectedPoseIndex if pose list shrinks
  useEffect(() => {
    if (playablePoses.length === 0) return
    if (selectedPoseIndex >= playablePoses.length) {
      setSelectedPoseIndex(playablePoses.length - 1)
    }
  }, [playablePoses.length, selectedPoseIndex, setSelectedPoseIndex])

  // ---- play loop driven by requestAnimationFrame ----
  const playStateRef = useRef({
    rafId: 0,
    lastTick: 0,
    accumulated: 0,
  })

  useEffect(() => {
    if (!isPlaying || playablePoses.length <= 1) return

    const tick = (now: number) => {
      const st = playStateRef.current
      if (st.lastTick === 0) st.lastTick = now
      const dt = (now - st.lastTick) / 1000
      st.lastTick = now
      st.accumulated += dt
      const frameDuration = 1 / fps
      if (st.accumulated >= frameDuration) {
        const advance = Math.floor(st.accumulated / frameDuration)
        st.accumulated -= advance * frameDuration
        const next = (useStore.getState().selectedPoseIndex + advance) % playablePoses.length
        setSelectedPoseIndex(next)
      }
      st.rafId = requestAnimationFrame(tick)
    }
    playStateRef.current.lastTick = 0
    playStateRef.current.accumulated = 0
    playStateRef.current.rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(playStateRef.current.rafId)
  }, [isPlaying, fps, playablePoses.length, setSelectedPoseIndex])

  const current = playablePoses[selectedPoseIndex] ?? null
  const sourcePoseMesh = current ? poseMeshes[current.pose_id] ?? null : null
  const resultMesh = current ? resultMeshes[current.pose_id] ?? null : null

  if (!session) return null

  // ---- Wrap / Region / Fit: single result (deformed source), not a pose
  // timeline. Left = source (before), right = result (after) with an optional
  // error heatmap. DT Transfer keeps the pose-animation view below. ----
  if (tool !== 'transfer') {
    if (!wrapMesh) {
      return (
        <div className="preview-empty">
          <h2>No result yet</h2>
          <p>
            Switch back to <b>Edit markers</b>, set up the tool, then press{' '}
            <b>Wrap</b> / <b>Fit</b>.
          </p>
        </div>
      )
    }
    const toolLabel =
      tool === 'region' ? 'Region' : tool === 'fit' ? 'Fit' : 'Wrap'
    const pct = (v: number) =>
      wrapMesh.residual
        ? `${((100 * v) / wrapMesh.residual.bbox_diag).toFixed(2)}%`
        : '—'
    return (
      <div className="preview-grid">
        <div className="viewports">
          <div className="viewport-cell">
            <MeshViewport
              title={
                session.source_ref
                  ? `Source (before): ${session.source_ref.name}`
                  : 'Source (before)'
              }
              mesh={sourceMesh}
              emptyHint="No source"
              meshColor={0x8fa6c4}
              wireframe={showWireframe}
            />
          </div>
          <div className="viewport-cell">
            {editingMask ? (
              <MeshViewport
                title="Edit body mask — L add · R remove"
                mesh={targetMesh}
                emptyHint="No body"
                meshColor={0x8fa6c4}
                paintMode
                onPaintStroke={onMaskPaint}
                frozenVerts={maskWorking}
                ghostMesh={wrapMesh}
                ghostStyle="ghost"
              />
            ) : (
              <MeshViewport
                title={`${toolLabel} result (after)`}
                mesh={wrapMesh}
                emptyHint="No result"
                wireframe={showWireframe}
                ghostMesh={bodyOn ? targetMesh : null}
                ghostStyle="solid"
                ghostMaskVerts={bodyMask?.hidden_vertices ?? null}
                ghostMaskApply={maskOn}
                heatValuesFrac={
                  showHeatmap
                    ? (heatMode === 'grip' && wrapMesh.grip_frac
                        ? wrapMesh.grip_frac
                        : wrapMesh.distances_frac ?? null)
                    : null
                }
                heatScaleFrac={heatMode === 'grip' && wrapMesh.grip_frac ? 1.0 : 0.02}
              />
            )}
          </div>
        </div>

        <aside className="sidebar">
          <section className="panel">
            <h3>{toolLabel} result</h3>
            <div className="hint">
              {wrapMesh.vertex_count.toLocaleString()} verts ·{' '}
              {wrapMesh.face_count.toLocaleString()} faces
            </div>
            {wrapMesh.residual && (
              <div className="hint" style={{ marginTop: 4 }}>
                Error vs target: mean <code>{pct(wrapMesh.residual.mean)}</code> · p95{' '}
                <code>{pct(wrapMesh.residual.p95)}</code> · max{' '}
                <code>{pct(wrapMesh.residual.max)}</code>
              </div>
            )}
            <label className="toggle" style={{ marginTop: 6 }}>
              <input
                type="checkbox"
                checked={showWireframe}
                onChange={(e) => setShowWireframe(e.target.checked)}
              />
              Show wireframe (topology)
            </label>
            {resultFromRefit && targetMesh && (
              <label
                className="toggle"
                title="Draw the target body under the fitted garment (solid — where the garment penetrates, you'll see it)."
              >
                <input
                  type="checkbox"
                  checked={showBody}
                  onChange={(e) => setShowBody(e.target.checked)}
                />
                Show body
              </label>
            )}
            {bodyOn && bodyMask && (
              <label
                className="toggle"
                title="Hide the body faces covered by the garment (the covered-skin mask computed by the fit)."
              >
                <input
                  type="checkbox"
                  checked={applyBodyMask}
                  onChange={(e) => setApplyBodyMask(e.target.checked)}
                />
                Apply body mask{' '}
                <span className="hint-inline">
                  ({bodyMask.hidden_vertices.length} of {bodyMask.body_vertex_count} verts)
                </span>
              </label>
            )}
            {resultFromRefit && targetMesh && bodyMask && !editingMask && (
              <div className="row" style={{ marginTop: 6, gap: 6 }}>
                <button
                  className="secondary"
                  style={{ width: 'auto' }}
                  onClick={enterMaskEdit}
                  title="Paint the covered-body mask on the body: left-drag adds, right-drag removes."
                >
                  ✎ Edit body mask
                </button>
                <a
                  className="link"
                  href={api.bodyMaskUrl(session.session_id)}
                  download
                  title="Download the mask as a JSON sidecar."
                >
                  .json
                </a>
              </div>
            )}
            {editingMask && (
              <div className="mask-editor" style={{ marginTop: 8 }}>
                <div className="hint">
                  Paint on the body: <b>left-drag</b> adds, <b>right-drag</b> removes.
                  The garment is shown faint for reference.
                </div>
                <div className="hint" style={{ marginTop: 4 }}>
                  hidden <code>{maskWorking.length.toLocaleString()}</code> of{' '}
                  <code>{bodyVertCount.toLocaleString()}</code> verts
                </div>
                <div className="row" style={{ marginTop: 6, gap: 6 }}>
                  <button className="secondary" style={{ width: 'auto' }}
                    disabled={maskBusy} onClick={() => doDilate(1)}
                    title="Grow the mask by one adjacency ring.">Dilate +1</button>
                  <button className="secondary" style={{ width: 'auto' }}
                    disabled={maskBusy} onClick={() => doDilate(-1)}
                    title="Shrink the mask by one adjacency ring.">Erode −1</button>
                  <button className="secondary" style={{ width: 'auto' }}
                    disabled={maskBusy || maskWorking === computedBaseline}
                    onClick={() => setMaskWorking(computedBaseline)}
                    title="Discard edits since you opened the editor.">Reset</button>
                </div>
                <div className="row" style={{ marginTop: 6, gap: 6 }}>
                  <button className="primary" style={{ width: 'auto' }}
                    disabled={maskBusy} onClick={doSaveMask}>
                    {maskBusy ? 'Saving…' : 'Save mask'}
                  </button>
                  <button className="secondary" style={{ width: 'auto' }}
                    disabled={maskBusy} onClick={() => setEditingMask(false)}>
                    Done
                  </button>
                </div>
              </div>
            )}
            {wrapMesh.distances_frac && (
              <label
                className="toggle"
                title="Color the result by distance to the target surface: blue = on it, red = ≥2% of its size. Red spots = add a marker there (or the target has no surface there)."
              >
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
                Heatmap
              </label>
            )}
            {showHeatmap && wrapMesh.grip_frac && (
              <div className="row" style={{ marginTop: 4, gap: 6 }}>
                {(['error', 'grip'] as const).map((m) => (
                  <button
                    key={m}
                    className={heatMode === m ? 'primary' : 'secondary'}
                    style={{ width: 'auto' }}
                    title={m === 'grip'
                      ? 'What the auto-detection decided: red = gripped onto the body, blue = left free (hanging cloth / released outer wall).'
                      : 'Distance to the body surface.'}
                    onClick={() => setHeatMode(m)}
                  >
                    {m === 'grip' ? 'Grip field' : 'Error'}
                  </button>
                ))}
              </div>
            )}
            {showHeatmap && wrapMesh.distances_frac && (
              <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{heatMode === 'grip' && wrapMesh.grip_frac ? 'free' : '0'}</span>
                <span
                  aria-hidden
                  style={{
                    flex: '0 1 140px',
                    height: 8,
                    borderRadius: 4,
                    background:
                      'linear-gradient(90deg, #3566d6 0%, #3fbf6f 50%, #e8d44a 75%, #e04a3a 100%)',
                  }}
                />
                <span>{heatMode === 'grip' && wrapMesh.grip_frac
                  ? 'gripped' : '≥2% of target size'}</span>
              </div>
            )}
          </section>

          {refitStats && (
            <section className="panel">
              <h3>Fit diagnostics</h3>
              {refitWarnings.length > 0 && (
                <div className="warnings" style={{ marginBottom: 6 }}>
                  {refitWarnings.map((w, i) => (
                    <div key={i} className="warning" style={{ marginTop: i ? 4 : 0 }}>
                      ⚠ {w}
                    </div>
                  ))}
                </div>
              )}
              <div className="hint" style={{ display: 'grid', gridTemplateColumns: 'auto auto', columnGap: 12, rowGap: 2 }}>
                {([
                  ['gripped (matched)', refitStats.matched_verts],
                  ['contact', refitStats.contact_verts],
                  ['seam grip', refitStats.seam_grip_verts],
                  ['occluded', refitStats.occluded_verts],
                  ['opposed normals', refitStats.opposed_verts],
                  ['frozen', refitStats.frozen_verts],
                  ['preserve (pins)', refitStats.preserve_verts],
                  ['body hidden', refitStats.hidden_body_verts],
                ] as [string, number | undefined][])
                  .filter(([, v]) => typeof v === 'number')
                  .map(([label, v]) => (
                    <div key={label} style={{ display: 'contents' }}>
                      <span>{label}</span>
                      <code style={{ textAlign: 'right' }}>{(v as number).toLocaleString()}</code>
                    </div>
                  ))}
              </div>
              {refitStats.bind && (
                <div className="hint subtle" style={{ marginTop: 4 }}>
                  ARAP bind: {refitStats.bind.handles ?? 0} handles ·{' '}
                  {refitStats.bind.free ?? 0} free · {refitStats.bind.iterations ?? 0} iters
                </div>
              )}
            </section>
          )}

          <section className="panel">
            <h3>Download</h3>
            <a className="primary as-button" href={api.wrapUrl(session.session_id)} download>
              Download .obj
            </a>
          </section>
        </aside>
      </div>
    )
  }

  if (playablePoses.length === 0) {
    return (
      <div className="preview-empty">
        <h2>No results yet</h2>
        <p>
          Switch back to <b>Edit markers</b>, set markers, upload poses, then press <b>Run</b>.
        </p>
      </div>
    )
  }

  return (
    <div className="preview-grid">
      <div className="viewports">
        <div className="viewport-cell">
          <MeshViewport
            title={current ? `Source pose: ${current.name}` : 'Source pose'}
            mesh={sourcePoseMesh}
            emptyHint={previewLoading ? 'Loading…' : 'No mesh'}
            meshColor={0x8fa6c4}
            wireframe={showWireframe}
          />
        </div>
        <div className="viewport-cell">
          <MeshViewport
            title={current ? `Deformed target: ${current.name}` : 'Deformed target'}
            mesh={resultMesh}
            emptyHint={previewLoading ? 'Loading…' : 'No result'}
            meshColor={0xc4a08f}
            wireframe={showWireframe}
          />
        </div>
      </div>

      <aside className="sidebar">
        <section className="panel">
          <h3>Playback</h3>
          <div className="row" style={{ marginBottom: 8 }}>
            <button className="primary" style={{ width: 'auto' }} onClick={togglePlay}>
              {isPlaying ? '❚❚  Pause' : '▶  Play'}
            </button>
            <span className="hint-inline">
              frame <code>{selectedPoseIndex + 1}</code> / {playablePoses.length}
            </span>
          </div>
          <label className="control">
            <span>
              Timeline:{' '}
              <code>{current?.name ?? ''}</code>
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, playablePoses.length - 1)}
              step={1}
              value={selectedPoseIndex}
              onChange={(e) => setSelectedPoseIndex(parseInt(e.target.value))}
            />
          </label>
          <label className="control">
            FPS: <b>{fps}</b>
            <input
              type="range" min={1} max={60} step={1}
              value={fps}
              onChange={(e) => setFps(parseInt(e.target.value))}
            />
          </label>
          <label className="toggle" style={{ marginTop: 4 }}>
            <input
              type="checkbox"
              checked={showWireframe}
              onChange={(e) => setShowWireframe(e.target.checked)}
            />
            Show wireframe (topology)
          </label>
          {previewLoading && <div className="hint subtle">Loading mesh data…</div>}
          {previewError && <div className="error">{previewError}</div>}
        </section>

        <section className="panel">
          <h3>Poses <span className="panel-count">({playablePoses.length})</span></h3>
          <ul className="pose-list pose-list-selectable">
            {playablePoses.map((p, i) => (
              <li
                key={p.pose_id}
                className={i === selectedPoseIndex ? 'selected' : ''}
                onClick={() => setSelectedPoseIndex(i)}
                title="click to jump to this pose"
              >
                <span className="seq">{String(i + 1).padStart(3, '0')}</span>
                <span>{p.name}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <h3>Download</h3>
          <a className="primary as-button" href={api.zipUrl(session.session_id)} download>
            Download all (.zip)
          </a>
          <a className="secondary as-button" href={api.fbxUrl(session.session_id)} download>
            Download .fbx (with shape keys)
          </a>
          {current && (
            <a
              className="link"
              href={api.resultUrl(session.session_id, current.pose_id)}
              download
            >
              Download current pose .obj
            </a>
          )}
        </section>
      </aside>
    </div>
  )
}
