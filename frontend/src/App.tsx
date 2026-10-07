import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, RefitPreset, RefitState } from './api'

/** One refit pin: a hard marker (inner source ↔ body, null if the needle didn't
 *  reach the body) plus the outer-wall verts to preserve/release. */
type RefitPinEntry = { marker: { source: number; target: number } | null; preserve: number[] }
import { useStore } from './store'
import MeshViewport, { markerColor, MeshViewportHandle } from './components/MeshViewport'
import PreviewPanel from './components/PreviewPanel'
import InfoPages from './InfoPages'
import { ToolSelect, HelpTip, StepGuide, StepKey, TOOLS } from './tools'
import { APP_VERSION } from './changelog'
import { meshBoundingRadius } from './util/mirror'

/** The viewport emits an identity placement when the gizmo is (re)built, so a
 *  non-null matrix does NOT mean the user moved anything. Only a matrix that
 *  differs from identity counts as "placed". */
function isPlacementSet(m: number[] | null): boolean {
  if (!m || m.length !== 16) return false
  return m.some((v, i) => Math.abs(v - (i % 5 === 0 ? 1 : 0)) > 1e-9)
}

export default function App() {
  const session = useStore((s) => s.session)
  const sourceMesh = useStore((s) => s.sourceMesh)
  const targetMesh = useStore((s) => s.targetMesh)
  const markers = useStore((s) => s.markers)
  const pendingPick = useStore((s) => s.pendingPick)
  const running = useStore((s) => s.running)
  const runError = useStore((s) => s.runError)
  const lastRun = useStore((s) => s.lastRun)
  const mode = useStore((s) => s.mode)
  const symmetryEnabled = useStore((s) => s.symmetryEnabled)
  const symmetryAxis = useStore((s) => s.symmetryAxis)
  const syncRotation = useStore((s) => s.syncRotation)
  const showWireframe = useStore((s) => s.showWireframe)
  const markerHistoryLen = useStore((s) => s.markerHistory.length)
  const markerRedoLen = useStore((s) => s.markerRedo.length)

  const initSession = useStore((s) => s.initSession)
  const restoreOrCreateSession = useStore((s) => s.restoreOrCreateSession)
  const uploadReference = useStore((s) => s.uploadReference)
  const refitViaProxy = useStore((s) => s.refitViaProxy)
  const uploadPoses = useStore((s) => s.uploadPoses)
  const deletePose = useStore((s) => s.deletePose)
  const pickVertex = useStore((s) => s.pickVertex)
  const removeMarker = useStore((s) => s.removeMarker)
  const clearMarkers = useStore((s) => s.clearMarkers)
  const addMarkerPair = useStore((s) => s.addMarkerPair)
  const resetPendingPick = useStore((s) => s.resetPendingPick)
  const run = useStore((s) => s.run)
  const wrap = useStore((s) => s.wrap)
  const refit = useStore((s) => s.refit)
  const previewGrip = useStore((s) => s.previewGrip)
  const clearGripPreview = useStore((s) => s.clearGripPreview)
  const gripPreview = useStore((s) => s.gripPreview)
  const gripLayer = useStore((s) => s.gripLayer)
  const gripPreviewing = useStore((s) => s.gripPreviewing)
  const wrapping = useStore((s) => s.wrapping)
  const wrapError = useStore((s) => s.wrapError)
  const polishMatrix = useStore((s) => s.polishMatrix)
  const wrapMesh = useStore((s) => s.wrapMesh)
  const dismissWrapError = useStore((s) => s.dismissWrapError)

  const regionTool = useStore((s) => s.regionTool)
  const regionWaypoints = useStore((s) => s.regionWaypoints)
  const regionClosed = useStore((s) => s.regionClosed)
  const regionSeed = useStore((s) => s.regionSeed)
  const regionInvert = useStore((s) => s.regionInvert)
  const setRegionInvert = useStore((s) => s.setRegionInvert)
  const regionLoop = useStore((s) => s.regionLoop)
  const regionEditable = useStore((s) => s.regionEditable)
  const regionPreviewInfo = useStore((s) => s.regionPreviewInfo)
  const regionPreviewing = useStore((s) => s.regionPreviewing)
  const regionError = useStore((s) => s.regionError)
  const setRegionTool = useStore((s) => s.setRegionTool)
  const closeRegionLoop = useStore((s) => s.closeRegionLoop)
  const undoRegionWaypoint = useStore((s) => s.undoRegionWaypoint)
  const clearRegion = useStore((s) => s.clearRegion)
  const previewRegion = useStore((s) => s.previewRegion)
  const dismissRegionError = useStore((s) => s.dismissRegionError)
  const jobStatus = useStore((s) => s.jobStatus)
  const tool = useStore((s) => s.tool)
  const setTool = useStore((s) => s.setTool)
  const setMode = useStore((s) => s.setMode)
  const setSymmetryEnabled = useStore((s) => s.setSymmetryEnabled)
  const setSymmetryAxis = useStore((s) => s.setSymmetryAxis)
  const setSyncRotation = useStore((s) => s.setSyncRotation)
  const setShowWireframe = useStore((s) => s.setShowWireframe)
  const undo = useStore((s) => s.undo)
  const redo = useStore((s) => s.redo)
  const exportMarkers = useStore((s) => s.exportMarkers)
  const importMarkers = useStore((s) => s.importMarkers)

  const [markerImportNote, setMarkerImportNote] = useState<{
    kind: 'ok' | 'err'
    text: string
  } | null>(null)
  const refInputMarkers = useRef<HTMLInputElement>(null)

  const onImportMarkersPick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    try {
      const r = await importMarkers(f)
      const parts = [`Added ${r.added}`]
      if (r.skipped) parts.push(`skipped ${r.skipped} dup`)
      if (r.invalid) parts.push(`${r.invalid} invalid`)
      setMarkerImportNote({
        kind: 'ok',
        text: parts.join(', ') + (r.warning ? ` — ⚠ ${r.warning}` : ''),
      })
    } catch (err: any) {
      setMarkerImportNote({ kind: 'err', text: err?.message ?? String(err) })
    }
  }

  const [iterations, setIterations] = useState(8)
  const [smoothness, setSmoothness] = useState(1.0)
  // Fit mode: identity weight as a log-scale "rigidity" knob. Higher = the
  // accessory keeps its shape and deformation localizes near the anchors.
  const [fitRigidityExp, setFitRigidityExp] = useState(-3) // 10^-3 = wrap default
  const [refitPreset, setRefitPreset] = useState<RefitPreset>('accessory')
  // Refit place/resize gizmo: which transform handle is active on the source
  // (garment), the resulting placement matrix (row-major 4x4), and a token we
  // bump to snap it back to identity.
  const [refitPlaceMode, setRefitPlaceMode] =
    useState<'off' | 'translate' | 'rotate' | 'scale'>('off')
  const [refitMatrix, setRefitMatrix] = useState<number[] | null>(null)
  const [refitResetToken, setRefitResetToken] = useState(0)
  // Auto-polish: nudge the placement to the contact standoff before the solve.
  const [snapPlacement, setSnapPlacement] = useState(false)
  // Refit pins ("needles"): poke through layers; inner grips, outer follows.
  const [refitPinMode, setRefitPinMode] = useState(false)
  const [refitPins, setRefitPins] = useState<RefitPinEntry[]>([])
  const [pinDepthFrac, setPinDepthFrac] = useState(0.12)
  const pinPreserveIdx = useMemo(() => refitPins.flatMap((p) => p.preserve), [refitPins])
  // A pin's inner grip is a real session marker; preserve verts ride refit state.
  const addPin = useCallback((pin: RefitPinEntry) => {
    if (pin.marker) addMarkerPair(pin.marker.source, pin.marker.target)
    setRefitPins((ps) => [...ps, pin])
  }, [addMarkerPair])
  const removePinMarker = useCallback((m: { source: number; target: number }) => {
    const cur = useStore.getState().markers
    const idx = cur.findIndex((x) => x.source === m.source && x.target === m.target)
    if (idx >= 0) useStore.getState().removeMarker(idx)
  }, [])
  const undoPin = useCallback(() => {
    setRefitPins((ps) => {
      const last = ps[ps.length - 1]
      if (last?.marker) removePinMarker(last.marker)
      return ps.slice(0, -1)
    })
  }, [removePinMarker])
  const clearPins = useCallback(() => {
    refitPins.forEach((p) => p.marker && removePinMarker(p.marker))
    setRefitPins([])
  }, [refitPins, removePinMarker])
  // Frozen paint: verts brushed as "do not deform at all". A Set for O(1)
  // stroke merging (strokes fire per pointer sample), array-ified for props.
  const [refitPaintMode, setRefitPaintMode] = useState(false)
  const [frozenSet, setFrozenSet] = useState<Set<number>>(() => new Set())
  const [paintRadiusFrac, setPaintRadiusFrac] = useState(0.04)
  // Layer auto-classification (L3): score-based driven/follower split. When on,
  // the grip dry-run also returns a per-vertex layer id (blue = follower wall,
  // purple = follower component) and enables "→ frozen paint" to prefill the
  // brush mask with the followers.
  const [layerAuto, setLayerAuto] = useState(false)
  // Advanced refit overrides (item 3b). World units; '' = preset default (sent
  // as null). Kept as strings so a blank field is unambiguous.
  const [contactTight, setContactTight] = useState('')
  const [contactFree, setContactFree] = useState('')
  const [thickness, setThickness] = useState('')
  const numOrNull = (s: string): number | null => {
    const v = parseFloat(s)
    return s.trim() !== '' && Number.isFinite(v) ? v : null
  }
  const refitOverrides = {
    contact_tight: numOrNull(contactTight),
    contact_free: numOrNull(contactFree),
    thickness: numOrNull(thickness),
  }
  const frozenIdx = useMemo(() => Array.from(frozenSet), [frozenSet])
  const onPaintStroke = useCallback((verts: number[], erase: boolean) => {
    setFrozenSet((prev) => {
      const next = new Set(prev)
      if (erase) verts.forEach((v) => next.delete(v))
      else verts.forEach((v) => next.add(v))
      return next.size === prev.size && !erase ? prev : next
    })
  }, [])
  // A grip preview is a snapshot of the current inputs — stale the moment any
  // of them change, so clear it (item 3c).
  useEffect(() => {
    clearGripPreview()
  }, [refitPreset, refitMatrix, pinPreserveIdx, frozenIdx,
      contactTight, contactFree, thickness, layerAuto, clearGripPreview])
  // If the source garment is (re)uploaded, the viewport rebuilds its mesh at
  // identity — drop any stale placement / pins so we don't send indices for a
  // mesh that's no longer there.
  const sourceRefName = session?.source_ref?.name
  useEffect(() => {
    setRefitPlaceMode('off')
    setRefitMatrix(null)
    setRefitPinMode(false)
    setRefitPins([])
    setRefitPaintMode(false)
    setFrozenSet(new Set())
  }, [sourceRefName])

  // ---- persist refit working state per session (item 3a) ----
  // Gate saving until we've hydrated this session, so the reset effect above
  // (and the initial mount) don't PUT empty state over a saved blob.
  const refitSaveReady = useRef(false)
  const sessionId = session?.session_id
  // Hydrate from the session summary on load. Runs AFTER the reset effect (both
  // fire on restore), so it wins. Only when the saved state was captured against
  // the CURRENT source — a changed garment discards it (indices would be stale).
  useEffect(() => {
    refitSaveReady.current = false
    const rs = session?.refit_state
    if (rs && rs.source_name && rs.source_name === sourceRefName) {
      setRefitMatrix(Array.isArray(rs.pre_transform) && rs.pre_transform.length === 16
        ? rs.pre_transform : null)
      if (Array.isArray(rs.pins)) setRefitPins(rs.pins)
      if (Array.isArray(rs.frozen)) setFrozenSet(new Set(rs.frozen))
      if (typeof rs.preset === 'string') setRefitPreset(rs.preset as RefitPreset)
      const adv = rs.advanced ?? { contact_tight: '', contact_free: '', thickness: '' }
      setContactTight(adv.contact_tight ?? '')
      setContactFree(adv.contact_free ?? '')
      setThickness(adv.thickness ?? '')
    }
    refitSaveReady.current = true
    // Depend only on the session identity: hydrate once per load, not on every
    // source-name churn (a source swap is handled by the reset effect above).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])
  // Debounced (1 s) PUT on any change to the working state.
  useEffect(() => {
    if (!sessionId || tool !== 'refit' || !refitSaveReady.current) return
    const blob: RefitState = {
      source_name: sourceRefName ?? null,
      pre_transform: refitMatrix,
      pins: refitPins,
      frozen: frozenIdx,
      preset: refitPreset,
      advanced: { contact_tight: contactTight, contact_free: contactFree, thickness },
    }
    const h = window.setTimeout(() => { api.putRefitState(sessionId, blob).catch(() => {}) }, 1000)
    return () => window.clearTimeout(h)
  }, [sessionId, tool, sourceRefName, refitMatrix, refitPins, frozenIdx,
      refitPreset, contactTight, contactFree, thickness])
  // Adopt the auto-polished placement into the gizmo once a refit reports it,
  // then clear the store flag so it fires exactly once per job.
  useEffect(() => {
    if (polishMatrix && polishMatrix.length === 16) {
      setRefitMatrix(polishMatrix)
      useStore.setState({ polishMatrix: null })
    }
  }, [polishMatrix])
  // Region feather as a fraction of the source's bounding radius, so the
  // seam softness is scale-independent across meshes.
  const [featherFrac, setFeatherFrac] = useState(0.05)
  // Seam sharpness: exponent of the feather falloff. 1 = gentle/linear,
  // higher = feather concentrates near the seam (firmer seam, wrap freedom
  // sooner inside). Matches RegionParams.feather_exponent (backend default 2).
  const [featherSharpness, setFeatherSharpness] = useState(2)
  // Align the target into the source's frame via markers before wrapping.
  // On by default — without it, meshes authored at different scales/positions
  // make the closest-point step collapse the result.
  const [alignToSource, setAlignToSource] = useState(true)
  // Taubin smoothing passes on the wrap/region result — removes the
  // high-frequency wobble the closest-point step injects. 0 = off.
  const [smoothPasses, setSmoothPasses] = useState(8)
  const [poseRejected, setPoseRejected] = useState<{ name: string; reason: string }[]>([])

  // ---- Advanced wrap knobs (surface matching). Defaults mirror the backend
  // WrapPayload defaults; a preset sets angle+distance+projection together.
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [matchAngle, setMatchAngle] = useState(60) // degrees
  const [matchDistFrac, setMatchDistFrac] = useState(0.02)
  const [projectResult, setProjectResult] = useState(true)
  const [projectDistFrac, setProjectDistFrac] = useState(0.02)
  const [softMarkers, setSoftMarkers] = useState(true)
  type Preset = 'loose' | 'normal' | 'tight' | 'custom'
  const [preset, setPreset] = useState<Preset>('normal')
  const applyPreset = (p: Preset) => {
    setPreset(p)
    if (p === 'loose') { setMatchAngle(75); setMatchDistFrac(0.04); setProjectDistFrac(0.04) }
    else if (p === 'normal') { setMatchAngle(60); setMatchDistFrac(0.02); setProjectDistFrac(0.02) }
    else if (p === 'tight') { setMatchAngle(45); setMatchDistFrac(0.01); setProjectDistFrac(0.01) }
    // 'custom' leaves the current values as-is
  }
  // Bundle the advanced knobs to spread into every wrap() call.
  const advancedWrapOpts = {
    match_max_angle_deg: matchAngle,
    match_distance_frac: matchDistFrac,
    project_result: projectResult,
    project_distance_frac: projectDistFrac,
    soft_markers: softMarkers,
  }
  const markerListRef = useRef<HTMLUListElement>(null)
  const lastMarkerCountRef = useRef(0)

  const lastFbxExtract = useStore((s) => s.lastFbxExtract)
  const uploadingRef = useStore((s) => s.uploadingRef)
  const uploadError = useStore((s) => s.uploadError)
  const dismissFbxExtract = useStore((s) => s.dismissFbxExtract)
  const dismissUploadError = useStore((s) => s.dismissUploadError)

  const sourceViewportRef = useRef<MeshViewportHandle>(null)
  const targetViewportRef = useRef<MeshViewportHandle>(null)
  const syncRotationRef = useRef(syncRotation)
  syncRotationRef.current = syncRotation

  const onSourceCameraChange = useCallback(() => {
    if (!syncRotationRef.current) return
    const state = sourceViewportRef.current?.getCameraState()
    if (state) targetViewportRef.current?.applyCameraState(state, true)
  }, [])
  const onTargetCameraChange = useCallback(() => {
    if (!syncRotationRef.current) return
    const state = targetViewportRef.current?.getCameraState()
    if (state) sourceViewportRef.current?.applyCameraState(state, true)
  }, [])

  // Ctrl-Z undo, Ctrl-Y / Ctrl-Shift-Z redo. Only when not focused in a text input.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null
      const tag = tgt?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tgt?.isContentEditable) return
      const ctrl = e.ctrlKey || e.metaKey
      if (!ctrl) return
      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        undo()
      } else if ((e.key === 'y') || (e.key === 'Z' && e.shiftKey)) {
        e.preventDefault()
        redo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  useEffect(() => {
    if (markers.length > lastMarkerCountRef.current && markerListRef.current) {
      markerListRef.current.scrollTop = markerListRef.current.scrollHeight
    }
    lastMarkerCountRef.current = markers.length
  }, [markers.length])

  useEffect(() => {
    if (!session) restoreOrCreateSession()
  }, [session, restoreOrCreateSession])

  // -------- session keep-alive (heartbeat) --------
  // Backend evicts sessions that haven't been touched for SESSION_TTL_SECONDS
  // (default 30 min). While this tab is open we ping every 5 min so an
  // idle-but-still-watching user doesn't get GC'd.
  //
  // We do NOT send a DELETE on beforeunload / pagehide. Those events fire on
  // page reload too, and we can't reliably tell reload from tab-close. A
  // reload-triggered DELETE would race with restoreOrCreateSession and the
  // user's localStorage-restored session would 404. Tab-closed sessions get
  // cleaned up by the TTL sweep within the heartbeat window — at most 30
  // minutes of orphan workspace, which is acceptable.
  useEffect(() => {
    if (!session) return
    const id = session.session_id

    const HEARTBEAT_MS = 5 * 60 * 1000
    const tick = () => {
      // Any successful GET to a session-scoped path counts — the backend's
      // _get_session helper bumps last_seen for us.
      fetch(`/api/sessions/${id}`).catch(() => {})
    }
    const handle = window.setInterval(tick, HEARTBEAT_MS)
    return () => window.clearInterval(handle)
  }, [session?.session_id])

  // Auto-jump to Preview after Run is handled inside the run() action so
  // restoring a session with prior results doesn't override the persisted mode.

  // Markers may only be picked on a step that actually SHOWS the marker list.
  // Otherwise a stray click on the placement step silently adds a pair the user
  // can neither see nor delete. Held in a ref because the pick callbacks are
  // stable and would close over a stale value.
  const canPickMarkersRef = useRef(false)
  const onPickSource = useCallback((v: number) => {
    // Region tools intercept source clicks: outline adds boundary waypoints,
    // seed sets the interior seed. Otherwise fall back to marker picking.
    const st = useStore.getState()
    if (st.regionTool === 'outline') st.addRegionWaypoint(v)
    else if (st.regionTool === 'seed') st.setRegionSeed(v)
    else if (canPickMarkersRef.current) pickVertex('source', v)
  }, [pickVertex])
  const onPickTarget = useCallback((v: number) => {
    // The region is a partition of SOURCE vertices (the output keeps source
    // topology), so the outline/seed tools are source-only. While a region
    // tool is active, ignore target clicks instead of dropping stray markers.
    if (useStore.getState().regionTool !== 'off') return
    if (canPickMarkersRef.current) pickVertex('target', v)
  }, [pickVertex])

  const sourceMarkedIndices = useMemo(() => markers.map((m) => m.source), [markers])
  const targetMarkedIndices = useMemo(() => markers.map((m) => m.target), [markers])

  const pendingSourceIdx = pendingPick?.side === 'source' ? pendingPick.vertexIdx : null
  const pendingTargetIdx = pendingPick?.side === 'target' ? pendingPick.vertexIdx : null

  const onDropSource = useCallback(
    async (f: File) => uploadReference('source_ref', f),
    [uploadReference],
  )
  const onDropTarget = useCallback(
    async (f: File) => uploadReference('target_ref', f),
    [uploadReference],
  )

  const refInputSource = useRef<HTMLInputElement>(null)
  const refInputTarget = useRef<HTMLInputElement>(null)
  const refInputProxy = useRef<HTMLInputElement>(null)
  const refInputPoses = useRef<HTMLInputElement>(null)

  const handlePosesFiles = async (files: File[]) => {
    if (files.length === 0) return
    const r = await uploadPoses(files)
    setPoseRejected(r.rejected)
  }

  const onPosesPickerChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    e.target.value = ''
    await handlePosesFiles(files)
  }

  const onPosesDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files ?? []).filter((f) => f.name.toLowerCase().endsWith('.obj'))
    await handlePosesFiles(files)
  }

  // Identical connectivity (e.g. target = a wrap of this very source): /run
  // skips correspondence entirely (identity triangle mapping) and needs NO
  // markers. Mirror the backend's check client-side so the Run gate and the
  // hints tell the user instead of demanding pointless markers.
  const sameTopology = useMemo(() => {
    if (!sourceMesh || !targetMesh) return false
    if (sourceMesh.vertex_count !== targetMesh.vertex_count) return false
    if (sourceMesh.face_count !== targetMesh.face_count) return false
    const a = sourceMesh.faces
    const b = targetMesh.faces
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false
    }
    return true
  }, [sourceMesh, targetMesh])

  // A target too dense for DT transfer disables Run (but Wrap still works).
  const dtAvailable = session?.target_dt_available !== false
  const canRun =
    !!session?.source_ref && !!session?.target_ref &&
    (markers.length >= 3 || sameTopology) &&
    (session?.poses.length ?? 0) >= 1 && dtAvailable && !running
  // Wrap reuses correspondence only — it doesn't need source poses since it
  // doesn't transfer per-pose deformations. On identical topology the
  // closest-point term (Wrap) / frozen pins (Region) register the meshes on
  // their own, so markers aren't required — same as same-topology Run. Fit
  // (no closest-point) always needs markers to anchor each island.
  const canWrap =
    !!session?.source_ref && !!session?.target_ref &&
    (markers.length >= 3 || sameTopology) && !wrapping
  const canFit =
    !!session?.source_ref && !!session?.target_ref && markers.length >= 3 && !wrapping
  // Refit needs no markers — rough placement + preset is enough. Markers, if
  // any, act as optional pins.
  const canRefit =
    !!session?.source_ref && !!session?.target_ref && !wrapping

  // ---- mode-first: current tool metadata + the step wizard ----
  // The sidebar shows ONE step at a time: the guide for it, then only the
  // controls that step needs. Step definitions (and all their copy) live in
  // tools.tsx; here we only decide which one is done and which one is showing.
  const toolMeta = TOOLS.find((t) => t.key === tool)
  const steps = toolMeta?.steps ?? []
  const hasResult = tool === 'transfer' ? !!lastRun : !!wrapMesh
  const refsReady = !!session?.source_ref && !!session?.target_ref
  const isStepDone = (key: StepKey): boolean => {
    switch (key) {
      case 'refs': return refsReady
      case 'poses': return (session?.poses.length ?? 0) >= 1
      case 'region': return regionClosed
      case 'place': return isPlacementSet(refitMatrix)
      // Refit's manual-correction step. Marker pairs are the main tool there,
      // so any of the three counts as work done.
      case 'refine': return markers.length > 0 || refitPins.length > 0 || frozenSet.size > 0
      case 'markers':
        return tool === 'fit' ? markers.length >= 3 : markers.length >= 3 || sameTopology
      case 'settings':
      case 'result': return hasResult
    }
  }

  const [stepIdx, setStepIdx] = useState(0)
  const clampedStepIdx = steps.length ? Math.min(stepIdx, steps.length - 1) : 0
  const activeStep = steps[clampedStepIdx]
  const activeKey = activeStep?.key

  // Landing on a tool: open the first thing that still needs doing, so a
  // restored session with meshes already in it doesn't start on "upload".
  useEffect(() => {
    if (!toolMeta) return
    // Only the INPUT steps decide where to land. 'settings'/'result' count as
    // done whenever the session holds any result, and a result produced by the
    // tool you just switched away from must not drop you on this tool's
    // result screen.
    const i = toolMeta.steps.findIndex(
      (s) => !s.optional && s.key !== 'settings' && s.key !== 'result' && !isStepDone(s.key),
    )
    const runIdx = toolMeta.steps.findIndex((s) => s.key === 'settings')
    setStepIdx(i >= 0 ? i : runIdx >= 0 ? runIdx : toolMeta.steps.length - 1)
    // Deliberately keyed on the tool alone: re-running this on every input
    // change would yank the user out of the step they're working in.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool])

  // Two auto-advances, both unambiguous: both meshes are in (nothing left to
  // do on the upload step) and a run finished (the result is what they want to
  // see). Everything else is driven by the Next/Back buttons.
  useEffect(() => {
    if (!refsReady) return
    setStepIdx((i) => (steps[i]?.key === 'refs' && i + 1 < steps.length ? i + 1 : i))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refsReady])
  const resultIdx = steps.findIndex((s) => s.key === 'result')
  // Keyed on a job FINISHING, not on "a result exists": restoring a session
  // also makes one appear, and that must not throw the user onto the result
  // screen of work they did yesterday.
  const wasBusyRef = useRef(false)
  useEffect(() => {
    const busy = running || wrapping
    if (wasBusyRef.current && !busy && hasResult && resultIdx >= 0) setStepIdx(resultIdx)
    wasBusyRef.current = busy
  }, [running, wrapping, hasResult, resultIdx])

  // Which steps own the marker list — the only ones where a click may create a
  // pair (see canPickMarkersRef). Refit has no markers step: pairs are an
  // override on Refine, alongside Poke and the freeze brush.
  const markersOnThisStep =
    activeKey === 'markers' || (tool === 'refit' && activeKey === 'refine')
  canPickMarkersRef.current = markersOnThisStep

  // Leaving a step turns its viewport mode off. Otherwise the outline tool (or
  // the poke needle, or the brush) keeps eating clicks two steps later and the
  // user has no idea why their marker isn't landing.
  useEffect(() => {
    if (activeKey !== 'region') setRegionTool('off')
    if (activeKey !== 'place') setRefitPlaceMode('off')
    if (activeKey !== 'refine') {
      setRefitPinMode(false)
      setRefitPaintMode(false)
    }
  }, [activeKey, setRegionTool])

  // ---- region (partial wrap) derived values ----
  const sourceRadius = useMemo(
    () => (sourceMesh ? meshBoundingRadius(sourceMesh.vertices) : 1),
    [sourceMesh],
  )
  const featherWidth = featherFrac * sourceRadius
  const canRegionWrap = canWrap && regionClosed
  const onRegionWrap = () =>
    wrap({
      iterations,
      smoothness,
      identity_weight: 0.001,
      use_closest_point: true,
      align_to_source: alignToSource,
      smooth_result: smoothPasses,
      ...advancedWrapOpts,
      region: {
        waypoints: regionWaypoints,
        seed: regionSeed,
        invert: regionInvert,
        feather_width: featherWidth,
        feather_exponent: featherSharpness,
      },
    })

  /**
   * User-facing status while a job is in flight, derived from the polled
   * jobStatus (exact queue_position + progress from the backend).
   * Returns null if not busy or nothing useful to say yet.
   */
  const STAGE_LABELS: Record<string, string> = {
    queued: 'Queued',
    precompute: 'Preparing',
    iterations: 'Solving',
    transfer: 'Transferring poses',
    projection: 'Projecting onto surface',
    export: 'Exporting',
    done: 'Finishing',
  }
  const queueMessage = ((): string | null => {
    if (!(running || wrapping)) return null
    const js = jobStatus
    if (!js) return 'Starting…'
    // Queued behind other jobs — report exact position.
    if (js.state === 'queued' && js.queue_position > 0) {
      return `In queue: ${js.queue_position} job${js.queue_position > 1 ? 's' : ''} ahead`
    }
    const p = js.progress
    const stage = STAGE_LABELS[p?.stage ?? ''] ?? 'Computing'
    if (p && p.total > 0) return `${stage} ${p.iter}/${p.total}`
    return `${stage}…`
  })()
  // Fraction 0..1 for a determinate progress bar, or null for indeterminate.
  const jobProgressFrac = ((): number | null => {
    const p = jobStatus?.progress
    if (!p || p.total <= 0) return null
    return Math.max(0, Math.min(1, p.iter / p.total))
  })()
  // Shared progress indicator (message + bar) shown under any Run/Wrap/Fit/
  // Region button while a job is in flight.
  const jobProgressEl =
    (running || wrapping) && queueMessage ? (
      <div className="hint" style={{ marginTop: 6 }}>
        <div>⏳ {queueMessage}</div>
        <div
          style={{
            height: 4,
            marginTop: 5,
            borderRadius: 2,
            overflow: 'hidden',
            background: 'var(--border, #333)',
          }}
        >
          <div
            style={{
              height: '100%',
              width: jobProgressFrac !== null ? `${Math.round(jobProgressFrac * 100)}%` : '100%',
              background: '#4c8bf5',
              opacity: jobProgressFrac !== null ? 1 : 0.4,
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      </div>
    ) : null

  // What a click does RIGHT NOW, shown in the viewport itself. The sidebar
  // explains the step; this is the reminder where the pointer already is.
  const sourceHud = ((): string | null => {
    if (activeKey === 'region' && regionTool === 'outline') {
      return regionWaypoints.length >= 3
        ? 'Click the magenta start point (or “Close loop”) to finish the outline'
        : 'Click points around the patch you want to change'
    }
    if (activeKey === 'region' && regionTool === 'seed') {
      return 'Click a vertex INSIDE the patch to mark which side is edited'
    }
    if (activeKey === 'place' && refitPlaceMode !== 'off') {
      return 'Drag a gizmo handle — the body is ghosted behind the garment'
    }
    if (activeKey === 'refine' && refitPinMode) {
      return 'Click the garment to poke a needle through it'
    }
    // Last, because an armed tool owns the click before markers do.
    if (markersOnThisStep && !refitPaintMode) {
      return pendingPick?.side === 'source'
        ? 'Now click the matching vertex on the RIGHT mesh'
        : 'Click a vertex to start a marker pair'
    }
    return null
  })()
  const targetHud =
    markersOnThisStep && !refitPaintMode && !refitPinMode
      ? pendingPick?.side === 'target'
        ? 'Now click the matching vertex on the LEFT mesh'
        : 'Click a vertex to start a marker pair'
      : null

  // Markers live on their own wizard step for most tools, but Refit has no
  // such step (its grip is automatic) — there they are one of the override
  // mechanisms on the Refine step. Same panel either way, so it's built once.
  const markerPanel = (
    <section className="panel">
      <h3>Marker pairs <span className="panel-count">({markers.length})</span></h3>
      <div className="hint subtle">
        Click a vertex on one mesh, then the matching one on the other.
        Right-click a placed dot to delete its pair.
      </div>
      {pendingPick !== null && (
        <div className="hint">
          <span className="pending">
            Waiting for the matching vertex on the{' '}
            <b>{pendingPick.side === 'source' ? 'right' : 'left'}</b> mesh — picked{' '}
            {pendingPick.side} <code>{pendingPick.vertexIdx}</code>{' '}
            <button className="link" onClick={resetPendingPick}>cancel</button>
          </span>
        </div>
      )}
      <div className="toggle-row">
        <label className="toggle">
          <input
            type="checkbox"
            checked={symmetryEnabled}
            onChange={(e) => setSymmetryEnabled(e.target.checked)}
          />
          Symmetric markers
        </label>
        <select
          className="axis-select"
          value={symmetryAxis}
          disabled={!symmetryEnabled}
          onChange={(e) => setSymmetryAxis(e.target.value as any)}
          title="Which coordinate to mirror (flip)"
        >
          <option value="X">X</option>
          <option value="Y">Y</option>
          <option value="Z">Z</option>
        </select>
      </div>
      <div className="toggle-row history-row">
        <button
          className="ghost"
          disabled={markerHistoryLen === 0}
          onClick={undo}
          title="Undo (Ctrl-Z)"
        >↶ Undo</button>
        <button
          className="ghost"
          disabled={markerRedoLen === 0}
          onClick={redo}
          title="Redo (Ctrl-Y)"
        >↷ Redo</button>
        <input
          ref={refInputMarkers}
          type="file"
          accept="application/json,.json"
          style={{ display: 'none' }}
          onChange={onImportMarkersPick}
        />
        <button
          className="ghost"
          onClick={() => refInputMarkers.current?.click()}
          title="Import markers from a JSON file"
        >⤓ Import</button>
        <button
          className="ghost"
          disabled={markers.length === 0}
          onClick={exportMarkers}
          title="Download current markers as a JSON file"
        >⤒ Export</button>
      </div>
      {markerImportNote && (
        <div className={markerImportNote.kind === 'err' ? 'error' : 'fbx-extract-note'} style={{ marginTop: 6 }}>
          <div>{markerImportNote.text}</div>
          <button className="link" onClick={() => setMarkerImportNote(null)}>dismiss</button>
        </div>
      )}
      {markers.length === 0 ? (
        <div className="empty">
          {tool === 'refit'
            ? 'No pairs yet — mark the interface (collar, hem, cuffs), or use Poke where the garment already sits close.'
            : sameTopology && tool !== 'fit'
            ? 'No markers needed — source and target share topology.'
            : 'No markers yet — at least 3 are required.'}
        </div>
      ) : (
        <ul className="marker-list" ref={markerListRef}>
          {markers.map((m, i) => (
            <li key={i}>
              <span className="swatch" style={{ background: markerColor(i) }} />
              <code>{m.source}</code>
              <span className="arrow">→</span>
              <code>{m.target}</code>
              <button className="x" onClick={() => removeMarker(i)} title="Remove">✕</button>
            </li>
          ))}
        </ul>
      )}
      {markers.length > 0 && (
        <div className="row">
          <button className="ghost" onClick={clearMarkers}>Clear all</button>
        </div>
      )}
    </section>
  )

  const sessionShort = session?.session_id?.slice(0, 6) ?? '...'

  // Live "session expires in …" label. Tick once a minute so the countdown
  // stays roughly current without a per-second re-render.
  const [nowSec, setNowSec] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const h = window.setInterval(() => setNowSec(Date.now() / 1000), 60_000)
    return () => window.clearInterval(h)
  }, [])

  // Show the changelog once after an update: if the browser last saw a
  // different (older) version, auto-open the "What's new" page. First-ever
  // visitors just get the version stamped silently.
  const [changelogOnMount, setChangelogOnMount] = useState<'changelog' | null>(null)
  useEffect(() => {
    const KEY = 'dt-ui:seenVersion'
    let seen: string | null = null
    try {
      seen = localStorage.getItem(KEY)
    } catch {
      /* private mode */
    }
    if (seen && seen !== APP_VERSION) setChangelogOnMount('changelog')
    try {
      localStorage.setItem(KEY, APP_VERSION)
    } catch {
      /* ignore */
    }
  }, [])
  const expiryLabel = ((): string | null => {
    const exp = session?.expires_at
    if (!exp) return null
    const rem = exp - nowSec
    if (rem <= 0) return 'expired'
    const h = Math.floor(rem / 3600)
    const m = Math.floor((rem % 3600) / 60)
    if (h >= 24) return `expires in ${Math.floor(h / 24)}d ${h % 24}h`
    if (h >= 1) return `expires in ${h}h ${m}m`
    if (m >= 1) return `expires in ${m}m`
    return 'expires in <1m'
  })()
  // The "Preview results" tab is available when EITHER a DT-transfer produced
  // per-pose results OR a Wrap/Region/Fit produced a wrap result.
  const hasResults = (session?.result_pose_ids.length ?? 0) > 0 || !!session?.has_wrap

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Chameleon Warp</div>
        <div className="mode-tabs">
          <button
            className={`mode-tab ${mode === 'edit' ? 'active' : ''}`}
            onClick={() => setMode('edit')}
          >
            Edit markers
          </button>
          <button
            className={`mode-tab ${mode === 'preview' ? 'active' : ''}`}
            disabled={!hasResults}
            onClick={() => setMode('preview')}
            title={hasResults ? 'View deformed results' : 'Run the pipeline first'}
          >
            Preview results{hasResults ? '' : ' (no results yet)'}
          </button>
        </div>
        <div className="topbar-info">
          <span>session: <code>{sessionShort}</code></span>
          {expiryLabel && (
            <span
              className="subtle"
              title="Sessions and their uploads are deleted automatically after this time."
              style={{ marginLeft: 8, opacity: 0.7 }}
            >
              · {expiryLabel}
            </span>
          )}
          <button
            className="ghost"
            onClick={() => {
              // Confirm only if there's actual user work to discard. A fresh
              // empty session has nothing worth protecting.
              const hasWork =
                !!session?.source_ref ||
                !!session?.target_ref ||
                (session?.poses.length ?? 0) > 0 ||
                markers.length > 0
              if (
                !hasWork ||
                window.confirm(
                  'Start a new session? Your current uploads, markers and results will be deleted from the server.',
                )
              ) {
                initSession()
              }
            }}
          >
            New session
          </button>
        </div>
      </header>

      {mode === 'preview' ? (
        <PreviewPanel />
      ) : tool === null ? (
        <ToolSelect onPick={setTool} />
      ) : (
        <main className="main-grid">
          <div className={'viewports' + (refitPinMode ? ' combined' : '')}>
            <div className="viewport-cell">
              <MeshViewport
                ref={sourceViewportRef}
                title={
                  session?.source_ref
                    ? `Source reference: ${session.source_ref.name}`
                    : 'Source reference'
                }
                mesh={sourceMesh}
                markedVertexIndices={sourceMarkedIndices}
                pendingVertex={pendingSourceIdx}
                emptyHint="Drop the SOURCE mesh here (.obj / .fbx) or use “Upload source” →"
                hudHint={sourceHud}
                onPickVertex={onPickSource}
                onRemoveMarker={removeMarker}
                onCancelPending={resetPendingPick}
                onDropFile={onDropSource}
                onUserCameraChange={onSourceCameraChange}
                wireframe={showWireframe}
                boundaryWaypoints={regionWaypoints}
                boundaryClosed={regionClosed}
                boundaryLoop={regionLoop ?? undefined}
                seedVertex={regionSeed}
                regionEditableIndices={regionEditable ?? undefined}
                transformMode={
                  tool === 'refit' && !refitPinMode && !refitPaintMode
                    ? refitPlaceMode : 'off'
                }
                onTransformChange={setRefitMatrix}
                transformResetToken={refitResetToken}
                initialTransform={tool === 'refit' ? refitMatrix : null}
                ghostMesh={
                  // On the placement step the body is the whole point of
                  // looking at the viewport — show it without making the user
                  // arm a gizmo first.
                  tool === 'refit' &&
                  (activeKey === 'place' || refitPlaceMode !== 'off' || refitPinMode)
                    ? targetMesh : null
                }
                pinMode={tool === 'refit' && refitPinMode && !refitPaintMode}
                pinDepthFrac={pinDepthFrac}
                pinPreserve={tool === 'refit' ? pinPreserveIdx : []}
                onAddPin={addPin}
                paintMode={tool === 'refit' && refitPaintMode}
                paintRadiusFrac={paintRadiusFrac}
                onPaintStroke={onPaintStroke}
                frozenVerts={tool === 'refit' ? frozenIdx : []}
                heatValuesFrac={tool === 'refit' && gripPreview ? gripPreview : null}
                heatScaleFrac={1}
                layerValues={tool === 'refit' && gripLayer ? gripLayer : null}
              />
            </div>
            {/* In pin mode the source viewport goes full-width and already shows
                the body overlaid, so the separate target panel is hidden. */}
            {!refitPinMode && (
            <div className="viewport-cell">
              <MeshViewport
                ref={targetViewportRef}
                title={
                  session?.target_ref
                    ? `Target reference: ${session.target_ref.name}`
                    : 'Target reference'
                }
                mesh={targetMesh}
                markedVertexIndices={targetMarkedIndices}
                pendingVertex={pendingTargetIdx}
                emptyHint="Drop the TARGET mesh here (.obj / .fbx) or use “Upload target” →"
                hudHint={targetHud}
                onPickVertex={onPickTarget}
                onRemoveMarker={removeMarker}
                onCancelPending={resetPendingPick}
                onDropFile={onDropTarget}
                onUserCameraChange={onTargetCameraChange}
                wireframe={showWireframe}
              />
            </div>
            )}
          </div>

          <aside className="sidebar">
            <section className="panel tool-header">
              <div className="tool-header-row">
                <button className="link" onClick={() => setTool(null)} title="Back to tool selection">
                  ← Tools
                </button>
                <h2 className="tool-title">
                  {toolMeta?.label}
                  {toolMeta?.beta && <span className="beta-badge inline">beta</span>}
                  {tool && <HelpTip tool={tool} />}
                </h2>
              </div>
              {/* Clickable stepper: it's a map of the workflow AND the way
                  back to a step you already passed. */}
              <ol className="stepper">
                {steps.map((s, i) => {
                  const done = isStepDone(s.key)
                  const cls = i === clampedStepIdx ? 'step current' : done ? 'step done' : 'step'
                  return (
                    <li key={s.key} className={cls}>
                      <button
                        className="step-btn"
                        onClick={() => setStepIdx(i)}
                        title={s.title}
                      >
                        <span className="step-dot">{done && i !== clampedStepIdx ? '✓' : i + 1}</span>
                        <span className="step-label">{s.label}</span>
                      </button>
                    </li>
                  )
                })}
              </ol>
              {/* Context that must survive step changes — otherwise step 4 has
                  no idea whether anything was ever uploaded. */}
              <div className="tool-status">
                <span className={session?.source_ref ? 'ok' : 'todo'} title={session?.source_ref?.name}>
                  {session?.source_ref ? '✓ source' : '· no source'}
                </span>
                <span className={session?.target_ref ? 'ok' : 'todo'} title={session?.target_ref?.name}>
                  {session?.target_ref ? '✓ target' : '· no target'}
                </span>
                <span className={markers.length > 0 ? 'ok' : 'todo'}>
                  {markers.length} marker{markers.length === 1 ? '' : 's'}
                </span>
                {tool === 'transfer' && (
                  <span className={(session?.poses.length ?? 0) > 0 ? 'ok' : 'todo'}>
                    {session?.poses.length ?? 0} pose{(session?.poses.length ?? 0) === 1 ? '' : 's'}
                  </span>
                )}
              </div>
            </section>

            {activeStep && (
              <StepGuide step={activeStep} index={clampedStepIdx} total={steps.length} />
            )}

            {activeKey === 'refs' && (
            <section className="panel">
              <h3>Meshes</h3>
              <div className="hint">
                Accepts <b>.obj</b> or <b>.fbx</b>. An FBX uploaded as <b>source</b> auto-extracts
                every shape key as a pose.
              </div>
              <input
                ref={refInputSource}
                type="file"
                accept=".obj,.fbx"
                style={{ display: 'none' }}
                onChange={async (e) => {
                  const f = e.target.files?.[0]
                  e.target.value = ''
                  if (f) await uploadReference('source_ref', f).catch(() => {})
                }}
              />
              <input
                ref={refInputTarget}
                type="file"
                accept=".obj,.fbx"
                style={{ display: 'none' }}
                onChange={async (e) => {
                  const f = e.target.files?.[0]
                  e.target.value = ''
                  if (f) await uploadReference('target_ref', f).catch(() => {})
                }}
              />
              <div className="row">
                <button
                  disabled={uploadingRef !== null}
                  onClick={() => refInputSource.current?.click()}
                >
                  {uploadingRef === 'source_ref'
                    ? 'Uploading…'
                    : session?.source_ref ? 'Replace source' : 'Upload source'}
                </button>
                <button
                  disabled={uploadingRef !== null}
                  onClick={() => refInputTarget.current?.click()}
                >
                  {uploadingRef === 'target_ref'
                    ? 'Uploading…'
                    : session?.target_ref ? 'Replace target' : 'Upload target'}
                </button>
              </div>
              <div className="row state">
                <span className={session?.source_ref ? 'ok' : 'todo'}>
                  {session?.source_ref ? `✓ ${session.source_ref.name}` : '· source missing'}
                  {session?.source_uploaded_as_fbx && <span className="fbx-badge">FBX</span>}
                </span>
                <span className={session?.target_ref ? 'ok' : 'todo'}>
                  {session?.target_ref ? `✓ ${session.target_ref.name}` : '· target missing'}
                  {session?.target_uploaded_as_fbx && <span className="fbx-badge">FBX</span>}
                </span>
              </div>
              {lastFbxExtract && (
                <div className="fbx-extract-note">
                  <div>
                    <b>Extracted {lastFbxExtract.extracted_poses.length} shape keys</b> from FBX
                    {' '}({lastFbxExtract.mesh_name}, {lastFbxExtract.vertex_count.toLocaleString()} verts)
                  </div>
                  <button className="link" onClick={dismissFbxExtract}>dismiss</button>
                </div>
              )}
              {uploadError && (
                <div className="error" style={{ marginTop: 8 }}>
                  {uploadError}
                  <button className="link" onClick={dismissUploadError} style={{ marginLeft: 8 }}>dismiss</button>
                </div>
              )}
            </section>
            )}

            {activeKey === 'markers' && markerPanel}

            {activeKey === 'poses' && (
            <section
              className="panel"
              onDragOver={(e) => e.preventDefault()}
              onDrop={onPosesDrop}
            >
              <h3>Source poses <span className="panel-count">({session?.poses.length ?? 0})</span></h3>
              <input
                ref={refInputPoses}
                type="file"
                accept=".obj"
                multiple
                style={{ display: 'none' }}
                onChange={onPosesPickerChange}
              />
              <div className="row">
                <button onClick={() => refInputPoses.current?.click()}>Choose .obj files…</button>
                <span className="hint-inline">or drop them on this panel</span>
              </div>
              {(session?.poses.length ?? 0) === 0 ? (
                <div className="empty">No poses yet.</div>
              ) : (
                <ul className="pose-list">
                  {session!.poses.map((p) => (
                    <li key={p.pose_id}>
                      <span title={p.pose_id}>{p.name}</span>
                      <button className="x" onClick={() => deletePose(p.pose_id)} title="Remove">✕</button>
                    </li>
                  ))}
                </ul>
              )}
              {poseRejected.length > 0 && (
                <div className="rejected">
                  <div className="rejected-title">Rejected:</div>
                  <ul>
                    {poseRejected.map((r, i) => (
                      <li key={i}><code>{r.name}</code>: {r.reason}</li>
                    ))}
                  </ul>
                  <button className="link" onClick={() => setPoseRejected([])}>dismiss</button>
                </div>
              )}
            </section>
            )}

            {activeKey === 'settings' && (
            <section className="panel">
              <h3>{toolMeta?.label} settings</h3>
              {/* Solver knobs apply to every tool but are rarely the thing that
                  needs changing — collapsed so the tool's own controls lead. */}
              <details className="adv-details">
                <summary>Solver — iterations {iterations}, smoothness {smoothness.toFixed(2)}</summary>
                <label className="control">
                  Iterations: <b>{iterations}</b>
                  <input
                    type="range" min={1} max={12} step={1}
                    value={iterations}
                    onChange={(e) => setIterations(parseInt(e.target.value))}
                  />
                </label>
                <label className="control">
                  Smoothness: <b>{smoothness.toFixed(2)}</b>
                  <input
                    type="range" min={0} max={5} step={0.05}
                    value={smoothness}
                    onChange={(e) => setSmoothness(parseFloat(e.target.value))}
                  />
                </label>
              </details>
              {tool === 'transfer' && (<>
              {sameTopology && (
                <div className="hint" style={{ color: '#4caf6a' }}>
                  ✓ Source and target have identical topology — direct transfer
                  (correspondence skipped), markers are not needed.
                </div>
              )}
              <button
                className="primary"
                disabled={!canRun}
                onClick={() => run({ iterations, smoothness, identity_weight: 0.001 })}
              >
                {running ? 'Running…' : 'Run (DT poses)'}
              </button>
              {running && jobProgressEl}
              {!dtAvailable && session?.target_ref && (
                <div className="hint" style={{ color: '#e0a54c' }}>
                  ⚠ Target is too dense for DT transfer (Run). Wrap still works —
                  or decimate the target and re-upload.
                </div>
              )}
              {!canRun && !running && (
                <div className="hint subtle">
                  {!session?.source_ref && '• source reference missing  '}
                  {!session?.target_ref && '• target reference missing  '}
                  {markers.length < 3 && !sameTopology && `• ${3 - markers.length} more markers needed  `}
                  {(session?.poses.length ?? 0) < 1 && '• no source poses  '}
                </div>
              )}
              {runError && <div className="error">Error: {runError}</div>}
              </>)}

              {tool === 'wrap' && (<>
              <label
                className="hint subtle"
                style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}
              >
                <input
                  type="checkbox"
                  checked={alignToSource}
                  onChange={(e) => setAlignToSource(e.target.checked)}
                />
                Align target to source via markers (scale/position) — recommended
              </label>
              <label className="control" style={{ marginTop: 2 }}>
                Smooth result (Taubin): <b>{smoothPasses}</b>
                <input
                  type="range" min={0} max={20} step={1}
                  value={smoothPasses}
                  onChange={(e) => setSmoothPasses(parseInt(e.target.value))}
                />
              </label>

              {/* Advanced surface-matching knobs + presets (collapsed by default) */}
              <button
                className="link"
                style={{ marginTop: 2, fontSize: 12 }}
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? '▾ Advanced' : '▸ Advanced'}
              </button>
              {showAdvanced && (
                <div className="advanced-box">
                  <div className="preset-row">
                    {(['loose', 'normal', 'tight'] as const).map((p) => (
                      <button
                        key={p}
                        className={preset === p ? 'preset active' : 'preset'}
                        onClick={() => applyPreset(p)}
                        title={
                          p === 'loose'
                            ? 'Permissive matching — for meshes with coverage gaps or bigger shape differences'
                            : p === 'tight'
                            ? 'Strict matching — for similar, well-aligned meshes'
                            : 'Balanced defaults'
                        }
                      >
                        {p[0].toUpperCase() + p.slice(1)}
                      </button>
                    ))}
                  </div>
                  <label className="control">
                    Match angle limit: <b>{matchAngle}°</b>
                    <input
                      type="range" min={10} max={90} step={5}
                      value={matchAngle}
                      onChange={(e) => { setMatchAngle(parseInt(e.target.value)); setPreset('custom') }}
                    />
                  </label>
                  <label className="control">
                    Match distance: <b>{(matchDistFrac * 100).toFixed(1)}%</b>
                    <input
                      type="range" min={0.5} max={8} step={0.5}
                      value={matchDistFrac * 100}
                      onChange={(e) => { setMatchDistFrac(parseFloat(e.target.value) / 100); setPreset('custom') }}
                    />
                  </label>
                  <label
                    className="hint subtle"
                    style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                  >
                    <input
                      type="checkbox"
                      checked={projectResult}
                      onChange={(e) => setProjectResult(e.target.checked)}
                    />
                    Final projection onto target surface
                  </label>
                  {projectResult && (
                    <label className="control">
                      Projection distance: <b>{(projectDistFrac * 100).toFixed(1)}%</b>
                      <input
                        type="range" min={0.5} max={8} step={0.5}
                        value={projectDistFrac * 100}
                        onChange={(e) => { setProjectDistFrac(parseFloat(e.target.value) / 100); setPreset('custom') }}
                      />
                    </label>
                  )}
                  <label
                    className="hint subtle"
                    style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                  >
                    <input
                      type="checkbox"
                      checked={softMarkers}
                      onChange={(e) => setSoftMarkers(e.target.checked)}
                    />
                    Soft markers (relax pins in the strong phase)
                  </label>
                </div>
              )}

              <button
                className="primary"
                style={{ marginTop: 8 }}
                disabled={!canWrap}
                onClick={() => wrap({ iterations, smoothness, identity_weight: 0.001, align_to_source: alignToSource, smooth_result: smoothPasses, ...advancedWrapOpts })}
              >
                {wrapping ? 'Wrapping…' : 'Wrap'}
              </button>
              {wrapping && jobProgressEl}
              {!canWrap && !wrapping && (
                <div className="hint subtle">
                  {!session?.source_ref && '• source reference missing  '}
                  {!session?.target_ref && '• target reference missing  '}
                  {markers.length < 3 && !sameTopology && `• ${3 - markers.length} more markers needed  `}
                </div>
              )}
              </>)}

              {/* Shared error block for Wrap, Fit AND Region — ungated so
                  whichever tool failed reports it next to its Run button. */}
              {wrapError && (
                <div className="error">
                  Error: {wrapError}{' '}
                  <button className="link" onClick={dismissWrapError}>dismiss</button>
                </div>
              )}

              {tool === 'fit' && (<>
              <label className="control">
                Rigidity: <b>10^{fitRigidityExp.toFixed(2)}</b>
                <input
                  type="range" min={-4} max={0} step={0.25}
                  value={fitRigidityExp}
                  onChange={(e) => setFitRigidityExp(parseFloat(e.target.value))}
                />
              </label>
              <button
                className="primary"
                style={{ marginTop: 8 }}
                disabled={!canFit}
                onClick={() =>
                  wrap({
                    iterations: 1,
                    smoothness,
                    identity_weight: 10 ** fitRigidityExp,
                    use_closest_point: false,
                    // Fit attaches an accessory in a SHARED space — the anchors
                    // must travel to the marker sockets. Marker-based alignment
                    // would cancel that displacement, so it's always off here.
                    align_to_source: false,
                  })
                }
              >
                {wrapping ? 'Fitting…' : 'Fit'}
              </button>
              {wrapping && jobProgressEl}
              {!canFit && !wrapping && (
                <div className="hint subtle" style={{ marginTop: 4 }}>
                  {!session?.source_ref && '• source (accessory) missing  '}
                  {!session?.target_ref && '• target missing  '}
                  {markers.length < 3 && `• ${3 - markers.length} more marker pairs needed  `}
                </div>
              )}
              </>)}
            </section>
            )}

            {/* ---- Refit: place / resize the garment on the body ---- */}
            {tool === 'refit' && activeKey === 'place' && (
            <section className="panel">
              <h3>Place / resize</h3>
              <div className="row" style={{ marginTop: 6, gap: 6, flexWrap: 'wrap' }}>
                {(([
                  ['translate', '⤧ Move'],
                  ['rotate', '⟳ Rotate'],
                  ['scale', '⤢ Scale'],
                ]) as ['translate' | 'rotate' | 'scale', string][]).map(([m, label]) => (
                  <button
                    key={m}
                    className={refitPlaceMode === m ? 'primary' : 'secondary'}
                    style={{ width: 'auto' }}
                    disabled={!session?.source_ref}
                    onClick={() => setRefitPlaceMode(refitPlaceMode === m ? 'off' : m)}
                  >
                    {label}
                  </button>
                ))}
                <button
                  className="secondary"
                  style={{ width: 'auto' }}
                  disabled={!isPlacementSet(refitMatrix) && refitPlaceMode === 'off'}
                  onClick={() => {
                    setRefitResetToken((t) => t + 1)
                    setRefitMatrix(null)
                  }}
                  title="Snap the garment back to its original position"
                >
                  ↺ Reset
                </button>
              </div>
              <div className="hint subtle" style={{ marginTop: 4 }}>
                {refitPlaceMode === 'off'
                  ? (isPlacementSet(refitMatrix)
                      ? 'Placement set ✓ — will be applied on Refit.'
                      : 'Not placed — fitting from where the garment already sits.')
                  : 'Drag the gizmo handles in the left viewport. Camera stays put while you drag.'}
              </div>
              <label
                className="hint"
                style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}
                title="Nudge the placement (small translate/rotate/scale) so the contact region sits at the cloth standoff before fitting. The adjusted placement is loaded back into the gizmo."
              >
                <input
                  type="checkbox"
                  checked={snapPlacement}
                  onChange={(e) => setSnapPlacement(e.target.checked)}
                />
                Snap placement to body
              </label>
            </section>
            )}

            {/* ---- Refit: the manual-correction step. Marker pairs lead, because
                   they are the main tool here; Poke below is the alternative
                   route to the same correspondences. ---- */}
            {tool === 'refit' && activeKey === 'refine' && markerPanel}

            {tool === 'refit' && activeKey === 'refine' && (
            <section className="panel">
              <h3>Preview grip</h3>
              <div className="hint subtle">
                A dry run: builds the grip field without solving, so you can see
                what the automation decided before you override anything.{' '}
                <b style={{ color: '#e04a3a' }}>Red</b> = gripped onto the body,{' '}
                <b style={{ color: '#3566d6' }}>blue</b> = left free.
              </div>
              <button
                className="secondary"
                style={{ marginTop: 8 }}
                disabled={!canRefit || gripPreviewing}
                title="Dry run: build the grip field only (no solve) and color the source — red = gripped onto the body, blue = left free."
                onClick={() => previewGrip({ preset: refitPreset, pre_transform: refitMatrix, preserve: pinPreserveIdx, frozen: frozenIdx, auto_polish: snapPlacement, layer_mode: layerAuto ? 'auto' : 'legacy', ...refitOverrides })}
              >
                {gripPreviewing ? 'Building grip…' : '◉ Preview grip'}
              </button>
              <label className="hint" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, cursor: 'pointer' }}
                title="Score-based driven/follower split (body-side voting + fan-out occlusion). Preview grip then colors followers blue (wall) / purple (component); use → frozen paint to prefill the brush with them.">
                <input type="checkbox" checked={layerAuto} onChange={(e) => setLayerAuto(e.target.checked)} />
                Layers: auto — also split driven walls from followers
              </label>
              {gripPreview && !gripLayer && (
                <div className="hint" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
                  <span>free</span>
                  <span aria-hidden style={{
                    flex: '0 1 120px', height: 8, borderRadius: 4,
                    background: 'linear-gradient(90deg, #3566d6 0%, #3fbf6f 50%, #e8d44a 75%, #e04a3a 100%)',
                  }} />
                  <span>gripped</span>
                </div>
              )}
              {gripLayer && (
                <div style={{ marginTop: 6 }}>
                  <div className="hint" style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    <span><span aria-hidden style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#b8c0cc', marginRight: 4 }} />driven</span>
                    <span><span aria-hidden style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#3566d6', marginRight: 4 }} />follower</span>
                    <span><span aria-hidden style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: '#9b59b6', marginRight: 4 }} />component</span>
                  </div>
                  <button
                    className="secondary"
                    style={{ marginTop: 6 }}
                    title="Add every follower (wall + component) vertex to the frozen-paint mask, so you can refine it with the brush instead of painting from scratch."
                    onClick={() => setFrozenSet((prev) => {
                      const next = new Set(prev)
                      gripLayer.forEach((l, i) => { if (l >= 1) next.add(i) })
                      return next
                    })}
                  >
                    → frozen paint ({gripLayer.reduce((n, l) => n + (l >= 1 ? 1 : 0), 0).toLocaleString()} verts)
                  </button>
                </div>
              )}

              <h4 style={{ margin: '16px 0 2px' }}>Pins (Poke)</h4>
              <div className="hint subtle">
                For a spot the grip missed. Poking a needle through the garment
                sets a <b>marker</b> where the inner layer meets the body (it grips
                there); any outer layer it pierces is marked{' '}
                <b style={{ color: '#ff8a3c' }}>preserve</b> — kept, never stuck
                to the body — so a double-wall collar keeps its standoff.
              </div>
              <div className="row" style={{ marginTop: 6, gap: 6, flexWrap: 'wrap' }}>
                <button
                  className={refitPinMode ? 'primary' : 'secondary'}
                  style={{ width: 'auto' }}
                  disabled={!session?.source_ref}
                  onClick={() => {
                    setRefitPinMode((v) => !v)
                    setRefitPlaceMode('off')   // pins and the gizmo are exclusive
                  }}
                >
                  📌 Poke{refitPinMode ? ' (on)' : ''}
                </button>
                <button
                  className="secondary"
                  style={{ width: 'auto' }}
                  disabled={refitPins.length === 0}
                  onClick={undoPin}
                >
                  ↶ Undo pin
                </button>
                <button
                  className="secondary"
                  style={{ width: 'auto' }}
                  disabled={refitPins.length === 0}
                  onClick={clearPins}
                >
                  Clear pins
                </button>
              </div>
              <label className="control" style={{ marginTop: 6 }}
                     title="How deep the needle reaches, as a fraction of the garment size. Longer catches more layers / reaches the body; shorter stays near the surface.">
                Needle depth: <b>{(pinDepthFrac * 100).toFixed(0)}%</b>
                <input
                  type="range" min={0.03} max={0.4} step={0.01}
                  value={pinDepthFrac}
                  onChange={(e) => setPinDepthFrac(parseFloat(e.target.value))}
                />
              </label>
              <div className="hint subtle" style={{ marginTop: 4 }}>
                Pins: <b>{refitPins.length}</b>
                {' · markers '}<b>{refitPins.filter((p) => p.marker).length}</b>
                {' · preserve '}<b>{pinPreserveIdx.length}</b>
                {refitPinMode && ' · click the garment to poke'}
              </div>

              <h4 style={{ margin: '16px 0 2px' }}>Freeze paint</h4>
              <div className="hint subtle">
                Brush over parts that must <b>not deform at all</b> (a collar,
                a buckle). Painted verts get no grip and ride the fit as one
                near-rigid piece. <b>Left-drag</b> paints,{' '}
                <b>right-drag</b> erases.
              </div>
              <div className="row" style={{ marginTop: 6, gap: 6, flexWrap: 'wrap' }}>
                <button
                  className={refitPaintMode ? 'primary' : 'secondary'}
                  style={{ width: 'auto' }}
                  disabled={!session?.source_ref}
                  onClick={() => {
                    setRefitPaintMode((v) => !v)
                    setRefitPinMode(false)      // paint, pins and gizmo are exclusive
                    setRefitPlaceMode('off')
                  }}
                >
                  ❄ Freeze paint{refitPaintMode ? ' (on)' : ''}
                </button>
                <button
                  className="secondary"
                  style={{ width: 'auto' }}
                  disabled={frozenSet.size === 0}
                  onClick={() => setFrozenSet(new Set())}
                >
                  Clear frozen
                </button>
              </div>
              <label className="control" style={{ marginTop: 6 }}
                     title="Brush radius as a fraction of the garment size.">
                Brush size: <b>{(paintRadiusFrac * 100).toFixed(0)}%</b>
                <input
                  type="range" min={0.01} max={0.12} step={0.005}
                  value={paintRadiusFrac}
                  onChange={(e) => setPaintRadiusFrac(parseFloat(e.target.value))}
                />
              </label>
              <div className="hint subtle" style={{ marginTop: 4 }}>
                Frozen: <b>{frozenSet.size}</b> verts
                {refitPaintMode && ' · L-drag paints, R-drag erases · middle-mouse orbits · Shift+` to fly'}
              </div>
            </section>
            )}

            {/* ---- Refit: material preset + the run button ---- */}
            {tool === 'refit' && activeKey === 'settings' && (
            <section className="panel">
              <h3>Material preset</h3>
              <div className="row" style={{ marginTop: 8, gap: 6, flexWrap: 'wrap' }}>
                {(([
                  ['accessory', 'Accessory'],
                  ['cloth', 'Cloth'],
                  ['armor', 'Armor'],
                  ['skintight', 'Skin-tight'],
                ]) as [RefitPreset, string][]).map(([key, label]) => (
                  <button
                    key={key}
                    className={refitPreset === key ? 'primary' : 'secondary'}
                    style={{ width: 'auto' }}
                    onClick={() => setRefitPreset(key)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="hint subtle" style={{ marginTop: 6 }}>
                {{
                  accessory: 'Grips the seam onto the body; the bulk keeps its shape (beard, fur trim).',
                  cloth: 'Drapes loosely; the interface grips and the body is pushed out where it pokes through the cloth.',
                  armor: 'Very stiff — keeps its form; the body underneath is hidden instead of inflating the armor.',
                  skintight: 'Conforms everywhere like a second skin (tight suit).',
                }[refitPreset]}
              </div>
              <details style={{ marginTop: 10 }}>
                <summary style={{ cursor: 'pointer' }}>Advanced (contact / thickness)</summary>
                <div className="hint subtle" style={{ marginTop: 6 }}>
                  Override the preset's contact band and cloth thickness, in world
                  units (the mesh's own scale). Leave blank to use the preset.
                </div>
                {(([
                  ['contact_tight', contactTight, setContactTight,
                    'Distance at which a garment vertex counts as fully gripping the body.'],
                  ['contact_free', contactFree, setContactFree,
                    'Distance beyond which a vertex is left free (no grip). Raise it to grip more.'],
                  ['thickness', thickness, setThickness,
                    'Standoff kept between the garment and the body (cloth thickness).'],
                ]) as [string, string, (v: string) => void, string][]).map(
                  ([key, val, setter, tip]) => (
                    <label className="control" key={key} style={{ marginTop: 6 }} title={tip}>
                      <span>{key} <span className="hint-inline">(world units)</span></span>
                      <input
                        type="number"
                        min={0}
                        step="0.001"
                        placeholder="preset"
                        value={val}
                        onChange={(e) => setter(e.target.value)}
                        style={{ width: 120 }}
                      />
                    </label>
                  ),
                )}
              </details>

              <button
                className="primary"
                style={{ marginTop: 12 }}
                disabled={!canRefit}
                onClick={() => refit({ preset: refitPreset, iterations, pre_transform: refitMatrix, preserve: pinPreserveIdx, frozen: frozenIdx, auto_polish: snapPlacement, layer_mode: layerAuto ? 'auto' : 'legacy', ...refitOverrides })}
              >
                {wrapping ? 'Refitting…' : 'Refit'}
              </button>
              {!canRefit && !wrapping && (
                <div className="hint subtle" style={{ marginTop: 4 }}>
                  Upload both a source (garment) and a target (body) first.
                </div>
              )}
              {wrapping && jobProgressEl}

              <details style={{ marginTop: 12 }}>
                <summary style={{ cursor: 'pointer' }}>Wear via proxy</summary>
                <div className="hint subtle" style={{ marginTop: 6 }}>
                  Fit the garment to a base character (proxy) once, then wear it on
                  any body: Wrap the proxy basemesh onto the body first, upload the
                  same basemesh below, then Run — the garment rides the wrap and a
                  short cleanup fits it. Placement comes from the binding (no gizmo).
                </div>
                <input
                  ref={refInputProxy}
                  type="file"
                  accept=".obj"
                  style={{ display: 'none' }}
                  onChange={async (e) => {
                    const f = e.target.files?.[0]
                    e.target.value = ''
                    if (f) await uploadReference('proxy_ref', f).catch(() => {})
                  }}
                />
                <div className="row" style={{ marginTop: 6 }}>
                  <button
                    disabled={uploadingRef !== null}
                    onClick={() => refInputProxy.current?.click()}
                  >
                    {uploadingRef === 'proxy_ref'
                      ? 'Uploading…'
                      : session?.proxy_ref ? 'Replace proxy basemesh' : 'Upload proxy basemesh'}
                  </button>
                </div>
                <div className="row state">
                  <span className={session?.proxy_ref ? 'ok' : 'todo'}>
                    {session?.proxy_ref ? `proxy: ${session.proxy_ref.name}` : 'no proxy basemesh'}
                  </span>
                  <span className={session?.has_wrap ? 'ok' : 'todo'}>
                    {session?.has_wrap ? 'wrap ready' : 'no wrap result'}
                  </span>
                </div>
                <button
                  className="secondary"
                  style={{ marginTop: 8 }}
                  disabled={!canRefit || !session?.proxy_ref || !session?.has_wrap}
                  onClick={() => refitViaProxy({ preset: refitPreset, iterations, frozen: frozenIdx })}
                >
                  {wrapping ? 'Fitting…' : 'Wear via proxy'}
                </button>
              </details>
            </section>
            )}

            {/* ---- Wrap Region: draw the patch that may move ---- */}
            {tool === 'region' && activeKey === 'region' && (
            <section className="panel">
              <h3>Outline the patch</h3>
              <div className="row" style={{ marginTop: 6 }}>
                <button
                  className={regionTool === 'outline' ? 'primary' : ''}
                  disabled={!sourceMesh}
                  onClick={() => setRegionTool(regionTool === 'outline' ? 'off' : 'outline')}
                >
                  ✎ Outline{regionTool === 'outline' ? ' (on)' : ''}
                </button>
                <button
                  className={regionTool === 'seed' ? 'primary' : ''}
                  disabled={!sourceMesh}
                  onClick={() => setRegionTool(regionTool === 'seed' ? 'off' : 'seed')}
                >
                  ◎ Seed{regionTool === 'seed' ? ' (on)' : ''}
                </button>
              </div>

              <div className="hint subtle" style={{ marginTop: 4 }}>
                Boundary points: <b>{regionWaypoints.length}</b>
                {regionWaypoints.length > 0 && regionWaypoints.length < 3 && ' (need ≥3)'}
                {' · '}
                {regionClosed
                  ? <span className="ok">loop closed ✓</span>
                  : (regionWaypoints.length >= 3 ? 'open — click start to close' : 'open')}
                {' · '}Seed: <b>{regionSeed !== null ? `#${regionSeed}` : 'auto (smaller side)'}</b>
                {regionInvert && <span> · <b>inverted</b></span>}
              </div>
              <div className="row" style={{ marginTop: 4 }}>
                <button
                  disabled={regionWaypoints.length < 3 || regionClosed}
                  onClick={closeRegionLoop}
                >
                  Close loop
                </button>
                <button
                  className={regionInvert ? 'primary' : ''}
                  disabled={!regionClosed}
                  onClick={() => setRegionInvert(!regionInvert)}
                  title="Edit the other side of the loop"
                >
                  ⇄ Invert
                </button>
                <button
                  disabled={regionWaypoints.length === 0 && !regionClosed}
                  onClick={undoRegionWaypoint}
                >
                  {regionClosed ? 'Reopen' : 'Undo point'}
                </button>
                <button
                  disabled={regionWaypoints.length === 0 && regionSeed === null}
                  onClick={clearRegion}
                >
                  Clear region
                </button>
              </div>

              <label className="control" style={{ marginTop: 6 }}>
                Feather (seam softness): <b>{(featherFrac * 100).toFixed(0)}%</b>
                <input
                  type="range" min={0} max={0.25} step={0.01}
                  value={featherFrac}
                  onChange={(e) => setFeatherFrac(parseFloat(e.target.value))}
                />
              </label>

              <label className="control" style={{ marginTop: 6 }}
                     title="How the feather falls off from the boundary inward. 1 = gentle/linear (feather spread across the whole band); higher = the feather concentrates near the seam, so the seam grips harder and the interior gains wrap freedom faster.">
                Seam sharpness: <b>{featherSharpness.toFixed(1)}</b>
                {featherSharpness <= 1.01 && <span className="subtle"> · gentle</span>}
                {featherSharpness >= 3.5 && <span className="subtle"> · sharp</span>}
                <input
                  type="range" min={1} max={5} step={0.5}
                  value={featherSharpness}
                  onChange={(e) => setFeatherSharpness(parseFloat(e.target.value))}
                />
              </label>

              <div className="row" style={{ marginTop: 4 }}>
                <button
                  disabled={!regionClosed || regionPreviewing}
                  onClick={() => previewRegion(featherWidth, featherSharpness)}
                >
                  {regionPreviewing ? 'Previewing…' : '◉ Preview region'}
                </button>
              </div>

              {regionPreviewInfo && (
                <div className="hint" style={{ marginTop: 4 }}>
                  Editable (green): <b>{regionPreviewInfo.interior_count.toLocaleString()}</b> ·
                  {' '}frozen: <b>{regionPreviewInfo.frozen_count.toLocaleString()}</b> ·
                  {' '}feather: <b>{regionPreviewInfo.feather_count.toLocaleString()}</b>
                  {' '}/ {regionPreviewInfo.total.toLocaleString()} verts
                  {regionPreviewInfo.interior_count > regionPreviewInfo.total * 0.5 && (
                    <div className="todo" style={{ marginTop: 2 }}>
                      ⚠ Most of the mesh is editable — if you meant the small patch,
                      click <b>⇄ Invert</b>.
                    </div>
                  )}
                </div>
              )}
              {regionError && (
                <div className="error">
                  Region: {regionError}{' '}
                  <button className="link" onClick={dismissRegionError}>dismiss</button>
                </div>
              )}
            </section>
            )}

            {/* ---- Wrap Region: the run button (the shaping lives one step back) ---- */}
            {tool === 'region' && activeKey === 'settings' && (
            <section className="panel">
              <h3>Wrap the region</h3>
              <div className="hint subtle">
                Seam: feather <b>{(featherFrac * 100).toFixed(0)}%</b>, sharpness{' '}
                <b>{featherSharpness.toFixed(1)}</b> —{' '}
                <button className="link" onClick={() => setStepIdx(steps.findIndex((s) => s.key === 'region'))}>
                  change
                </button>
              </div>
              <button
                className="primary"
                style={{ marginTop: 8 }}
                disabled={!canRegionWrap}
                onClick={onRegionWrap}
              >
                {wrapping ? 'Wrapping…' : 'Wrap region'}
              </button>
              {wrapping && jobProgressEl}
              {!canRegionWrap && !wrapping && (
                <div className="hint subtle" style={{ marginTop: 4 }}>
                  {!regionClosed && '• the outline loop is not closed yet  '}
                  {markers.length < 3 && !sameTopology && `• ${3 - markers.length} more marker pairs needed  `}
                </div>
              )}
              {regionError && (
                <div className="error">
                  Region: {regionError}{' '}
                  <button className="link" onClick={dismissRegionError}>dismiss</button>
                </div>
              )}
            </section>
            )}

            {/* ---- Result: whatever this tool produced ---- */}
            {activeKey === 'result' && wrapMesh && session && (
              <section className="panel">
                <h3>Result</h3>
                <div className="hint">
                  {wrapMesh.vertex_count.toLocaleString()} verts ·{' '}
                  {wrapMesh.face_count.toLocaleString()} faces
                  {wrapMesh.residual && (
                    // Distances as % of the target's bbox diagonal — scale-free,
                    // comparable across runs. p95 is the number to tune against.
                    <div style={{ marginTop: 4 }}>
                      Error vs target: mean{' '}
                      <code>{(100 * wrapMesh.residual.mean / wrapMesh.residual.bbox_diag).toFixed(2)}%</code>{' '}
                      · p95{' '}
                      <code>{(100 * wrapMesh.residual.p95 / wrapMesh.residual.bbox_diag).toFixed(2)}%</code>{' '}
                      · max{' '}
                      <code>{(100 * wrapMesh.residual.max / wrapMesh.residual.bbox_diag).toFixed(2)}%</code>
                    </div>
                  )}
                </div>
                <button className="primary" onClick={() => setMode('preview')}>
                  Open preview →
                </button>
                <a className="secondary as-button" href={api.wrapUrl(session.session_id)} download>
                  Download .obj
                </a>
              </section>
            )}

            {activeKey === 'result' && !hasResult && (
              <section className="panel">
                <div className="empty">
                  Nothing to show yet — go back to{' '}
                  <button className="link" onClick={() => setStepIdx(Math.max(0, resultIdx - 1))}>
                    the previous step
                  </button>{' '}
                  and run the tool.
                </div>
              </section>
            )}

            {activeKey === 'result' && lastRun && session && (
              <section className="panel">
                <h3>Results</h3>
                <div className="hint">
                  {lastRun.identity_mapping
                    ? 'Direct transfer (identical topology, no correspondence) · '
                    : lastRun.mapping_size > 0
                    ? `Mapping size: ${lastRun.mapping_size} triangle pairs · `
                    : 'Restored from previous session · '}
                  {lastRun.results.length} pose(s) transferred
                </div>
                <button
                  className="primary"
                  onClick={() => setMode('preview')}
                >
                  Open Preview
                </button>
                <a className="primary as-button" href={api.zipUrl(session.session_id)} download>
                  Download all (.zip)
                </a>
                <a className="secondary as-button" href={api.fbxUrl(session.session_id)} download>
                  Download .fbx (with shape keys)
                </a>
                <ul className="result-list">
                  {lastRun.results.map((r) => (
                    <li key={r.pose_id}>
                      <span>{r.name}</span>
                      <a
                        href={api.resultUrl(session.session_id, r.pose_id)}
                        download
                        className="link"
                      >download .obj</a>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Viewport preferences — same on every step, so they live outside
                the wizard flow instead of hiding inside one of its steps. */}
            <section className="panel view-panel">
              <div className="toggle-row">
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={showWireframe}
                    onChange={(e) => setShowWireframe(e.target.checked)}
                  />
                  Show wireframe
                </label>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={syncRotation}
                    onChange={(e) => setSyncRotation(e.target.checked)}
                  />
                  Sync rotation
                </label>
              </div>
            </section>
          </aside>

          {/* Wizard navigation, pinned so it never scrolls out of reach. */}
          <nav className="step-nav">
            <button
              className="ghost"
              disabled={clampedStepIdx === 0}
              onClick={() => setStepIdx(clampedStepIdx - 1)}
            >
              ← Back
            </button>
            <span className="step-nav-label">
              {activeStep ? `${activeStep.label}${activeStep.optional ? ' (optional)' : ''}` : ''}
            </span>
            <button
              className="primary"
              disabled={clampedStepIdx >= steps.length - 1}
              onClick={() => setStepIdx(clampedStepIdx + 1)}
              title={
                activeStep && !activeStep.optional && !isStepDone(activeStep.key)
                  ? 'This step still looks unfinished — you can continue anyway.'
                  : undefined
              }
            >
              Next: {steps[clampedStepIdx + 1]?.label ?? '—'} →
            </button>
          </nav>
        </main>
      )}
      <InfoPages openOnMount={changelogOnMount} />
    </div>
  )
}
