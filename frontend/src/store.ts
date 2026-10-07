import { create } from 'zustand'
import { api, BodyMask, FbxExtractInfo, JobStatus, MarkerPair, MeshData, QueueStatus, RefitOpts, RefitStats, RegionPreview, Role, RunResponse, SessionSummary, WrapOpts, WrapResponse } from './api'
import { defaultMirrorTolerance, findMirrorVertex, meshBoundingRadius, meshCentroid } from './util/mirror'

export type Mode = 'edit' | 'preview'
export type Side = 'source' | 'target'
export type MirrorAxis = 'X' | 'Y' | 'Z'
// Which tool the user picked on the mode-first landing screen. null = show the
// tool-selection screen. Persisted so a reload lands back in the same tool.
export type ToolKind = 'wrap' | 'region' | 'transfer' | 'fit' | 'refit'

// --- localStorage keys for per-browser persistence -------------------------

const LS_SESSION_ID = 'dt-ui:sessionId'
const LS_PREFS = 'dt-ui:prefs'

// In-flight preview-mesh fetches, shared across concurrent loadPreviewMeshes
// passes. Keyed by kind:sessionId:poseId; entries remove themselves on settle.
const meshFetchInflight = new Map<string, Promise<void>>()

interface PersistedPrefs {
  symmetryEnabled?: boolean
  symmetryAxis?: MirrorAxis
  syncRotation?: boolean
  showWireframe?: boolean
  mode?: Mode
  fps?: number
  tool?: ToolKind | null
}

function loadPrefs(): PersistedPrefs {
  try {
    const raw = localStorage.getItem(LS_PREFS)
    if (!raw) return {}
    return JSON.parse(raw) as PersistedPrefs
  } catch {
    return {}
  }
}

function savePrefs(p: PersistedPrefs) {
  try {
    localStorage.setItem(LS_PREFS, JSON.stringify(p))
  } catch {
    /* quota exceeded, private mode, etc. — silently give up */
  }
}

function saveSessionId(id: string | null) {
  try {
    if (id === null) localStorage.removeItem(LS_SESSION_ID)
    else localStorage.setItem(LS_SESSION_ID, id)
  } catch {
    /* ignore */
  }
}

function loadSessionId(): string | null {
  try {
    return localStorage.getItem(LS_SESSION_ID)
  } catch {
    return null
  }
}

const INITIAL_PREFS = loadPrefs()

export interface SortedPose {
  pose_id: string
  name: string
}

export interface PendingPick {
  side: Side
  vertexIdx: number
}

const MAX_HISTORY = 100

interface State {
  // ---- session state ----
  session: SessionSummary | null
  sourceMesh: MeshData | null
  targetMesh: MeshData | null
  markers: MarkerPair[]
  pendingPick: PendingPick | null
  running: boolean
  runError: string | null
  lastRun: {
    mapping_size: number
    identity_mapping?: boolean
    results: { pose_id: string; name: string }[]
  } | null

  // ---- wrap mode (independent of /run; just runs correspondence) ----
  wrapping: boolean
  wrapError: string | null
  // Refit auto-polish: when a refit with auto_polish fires, the effective
  // placement (row-major 4x4) rides back here so the gizmo can adopt it.
  // App.tsx consumes it into refitMatrix and clears it. null = no update.
  polishMatrix: number[] | null
  wrapMesh: MeshData | null
  // True when the current wrapMesh came from a REFIT (not a plain wrap/region/
  // fit). Drives the combined "garment on the body" preview: the body toggle
  // defaults ON only for refit results.
  resultFromRefit: boolean
  // The covered-body-vertex mask that ships with a refit result (item 1/2):
  // which target verts sit under the fitted garment. null when the result
  // isn't a refit or has no mask yet.
  bodyMask: BodyMask | null
  // Refit result diagnostics for the stats block + warnings (item 3b). Only
  // set for a fresh refit result (not persisted across restore). null otherwise.
  refitStats: RefitStats | null
  // "Preview grip" dry-run (item 3c): per-source-vertex grip field to color the
  // source viewport (0 = free, 1 = gripped). null = not shown. Auto-clears on
  // any refit input change.
  gripPreview: number[] | null
  gripPreviewing: boolean
  // Auto layer mode (L3): per-source-vertex layer id from the dry-run
  // (0 driven / 1 follower-wall / 2 follower-component), used to color the
  // source viewport and to prefill the frozen-paint mask. null = not shown.
  gripLayer: number[] | null
  // Color the wrap result by per-vertex distance to the target surface
  // (blue = on it, red = far), shown in the Preview results tab. Only
  // effective when the wrap mesh carries distances_frac (Wrap/Region, not Fit).
  showHeatmap: boolean

  // ---- region (partial/masked wrap) drawing on the SOURCE viewport ----
  // regionTool gates what a source-viewport click does: add a boundary
  // waypoint, set the interior seed, or nothing (normal marker picking).
  regionTool: 'off' | 'outline' | 'seed'
  regionWaypoints: number[]            // clicked boundary points, in order
  regionClosed: boolean                // true once the loop is closed (back to start)
  regionSeed: number | null            // optional vertex on the editable side
  regionInvert: boolean                // flip which side of the loop is editable
  regionLoop: number[] | null          // dense stitched loop (from preview)
  regionEditable: number[] | null      // interior vertex indices (from preview)
  regionPreviewInfo: RegionPreview | null
  regionPreviewing: boolean
  regionError: string | null

  // ---- mode-first tool selection ----
  // Which tool the user picked. null → show the tool-selection landing screen.
  tool: ToolKind | null

  // ---- compute queue feedback ----
  // Snapshot of the global compute queue. Updated by a poller that runs
  // while `running` or `wrapping` is true. Null when we're not watching.
  queueStatus: QueueStatus | null
  // Live status of the current background job (/run or /wrap): state,
  // queue_position, and progress {stage, iter, total}. Updated by the job
  // poller while running/wrapping; null when idle. Drives the progress bar
  // and "N jobs ahead" message at the Run/Wrap buttons.
  jobStatus: JobStatus | null

  // ---- preview / animation state ----
  mode: Mode
  selectedPoseIndex: number
  poseMeshes: Record<string, MeshData>
  resultMeshes: Record<string, MeshData>
  previewLoading: boolean
  previewError: string | null
  isPlaying: boolean
  fps: number

  // ---- last FBX extract ----
  lastFbxExtract: FbxExtractInfo | null
  uploadingRef: Role | null
  uploadError: string | null

  // ---- UX toggles ----
  symmetryEnabled: boolean
  symmetryAxis: MirrorAxis        // which coordinate to flip when mirroring
  syncRotation: boolean
  showWireframe: boolean          // overlay the triangle topology in viewports

  // ---- markers undo / redo history (snapshots of the markers array) ----
  markerHistory: MarkerPair[][]   // snapshots BEFORE each change, oldest first
  markerRedo: MarkerPair[][]      // snapshots that can be redone

  // ---- actions ----
  initSession: () => Promise<void>
  restoreOrCreateSession: () => Promise<void>
  refreshSession: () => Promise<void>
  uploadReference: (role: Role, file: File) => Promise<void>
  uploadPoses: (files: File[]) => Promise<{ rejected: { name: string; reason: string }[] }>
  deletePose: (poseId: string) => Promise<void>
  pickVertex: (side: Side, vertexIdx: number) => void
  /** Add a complete source→target marker pair directly (used by refit pins,
   *  which poke a needle through both meshes at once). No-op on exact dup. */
  addMarkerPair: (source: number, target: number) => void
  removeMarker: (idx: number) => void
  clearMarkers: () => void
  resetPendingPick: () => void
  syncMarkers: () => Promise<void>
  exportMarkers: () => void
  importMarkers: (file: File) => Promise<{
    added: number
    skipped: number
    invalid: number
    warning?: string
  }>
  run: (opts: { iterations: number; smoothness: number; identity_weight: number }) => Promise<void>
  wrap: (opts: WrapOpts) => Promise<void>
  refit: (opts: RefitOpts) => Promise<void>
  refitViaProxy: (opts: RefitOpts) => Promise<void>
  // Re-attach to a job that's still running server-side (e.g. the tab was
  // closed mid-compute and reopened). Called from session restore.
  resumeJobIfActive: () => Promise<void>
  setShowHeatmap: (v: boolean) => void
  dismissWrapError: () => void
  // Body-mask editing (item 2). Both persist server-side and update bodyMask;
  // dilate first PUTs the working set so it grows/shrinks what's on screen.
  saveBodyMask: (hidden: number[]) => Promise<BodyMask | null>
  dilateBodyMask: (rings: number, hidden: number[]) => Promise<BodyMask | null>
  previewGrip: (opts: RefitOpts) => Promise<void>
  clearGripPreview: () => void

  setRegionTool: (t: 'off' | 'outline' | 'seed') => void
  addRegionWaypoint: (v: number) => void
  closeRegionLoop: () => void
  undoRegionWaypoint: () => void
  clearRegion: () => void
  setRegionSeed: (v: number) => void
  setRegionInvert: (v: boolean) => void
  previewRegion: (featherWidth: number, featherExponent?: number) => Promise<void>
  dismissRegionError: () => void

  setTool: (tool: ToolKind | null) => void
  setMode: (mode: Mode) => void
  setSelectedPoseIndex: (idx: number) => void
  loadPreviewMeshes: (priorityPoseId?: string) => Promise<void>
  togglePlay: () => void
  setFps: (fps: number) => void
  dismissFbxExtract: () => void
  dismissUploadError: () => void

  setSymmetryEnabled: (v: boolean) => void
  setSymmetryAxis: (a: MirrorAxis) => void
  setSyncRotation: (v: boolean) => void
  setShowWireframe: (v: boolean) => void

  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
}

export const sortPoses = (poses: { pose_id: string; name: string }[]): SortedPose[] =>
  [...poses].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }))

const axisToIndex = (a: MirrorAxis): 0 | 1 | 2 => (a === 'X' ? 0 : a === 'Y' ? 1 : 2)

/**
 * Push the current markers state into history before applying a mutation, so
 * the change can be undone. Any pending redo tail is wiped (standard undo
 * semantics — once you make a new change, the redo stack is invalidated).
 */
function pushHistory(state: { markers: MarkerPair[]; markerHistory: MarkerPair[][]; markerRedo: MarkerPair[][] }) {
  const next = [...state.markerHistory, [...state.markers]]
  if (next.length > MAX_HISTORY) next.shift()
  return { markerHistory: next, markerRedo: [] as MarkerPair[][] }
}

/**
 * Poll GET /sessions/{id}/job at a 1-second cadence until the job reaches a
 * terminal state (done/error), pushing each snapshot into `jobStatus` so the
 * UI can render progress + queue position. Resolves with the terminal
 * JobStatus, or null if we lost track of it (session changed, or the job
 * vanished — e.g. the backend restarted mid-compute).
 *
 * `isAlive` lets the caller bail if the user switched sessions underneath us.
 */
function pollJob(
  sessionId: string,
  set: (patch: any) => void,
  isAlive: () => boolean,
): Promise<JobStatus | null> {
  return new Promise((resolve) => {
    let stopped = false
    let misses = 0
    let handle = 0
    const finish = (job: JobStatus | null) => {
      if (stopped) return
      stopped = true
      window.clearInterval(handle)
      resolve(job)
    }
    const tick = async () => {
      if (stopped) return
      if (!isAlive()) return finish(null)
      try {
        const job = await api.getJob(sessionId)
        if (!job) {
          // 404 — the job hasn't registered yet, or the backend restarted and
          // lost it. Tolerate a few misses before giving up.
          if (++misses > 5) return finish(null)
          return
        }
        misses = 0
        set({ jobStatus: job })
        if (job.state === 'done' || job.state === 'error') return finish(job)
      } catch {
        /* transient network/backend hiccup — keep polling */
      }
    }
    handle = window.setInterval(tick, 1000)
    void tick()
  })
}

/** Apply a finished /run job's result to the store (shared by run() and the
 * resume-on-restore path). Jumps to the preview tab and kicks off mesh load. */
function applyRunDone(r: RunResponse, set: (patch: any) => void, get: () => State) {
  set({
    running: false,
    jobStatus: null,
    session: r.session,
    lastRun: {
      mapping_size: r.mapping_size,
      identity_mapping: r.identity_mapping,
      results: r.results,
    },
    resultMeshes: {},
    selectedPoseIndex: 0,
    isPlaying: false,
    mode: 'preview',
  })
  savePrefs({ ...loadPrefs(), mode: 'preview' })
  get().loadPreviewMeshes()
}

/** Apply a finished /wrap job's result: fetch the wrap mesh, show it, and
 * (for a fresh wrap, not a resume) auto-download the OBJ. */
async function applyWrapDone(
  sessionId: string,
  r: WrapResponse,
  set: (patch: any) => void,
  opts: { download: boolean; targetMesh?: MeshData | null },
) {
  const fromRefit = !!r.refit
  // A refit result gets the combined preview (garment ON the body): fetch the
  // covered-body mask, and make sure the body mesh is available even in a
  // preview-only flow where the editor never loaded target_ref.
  const [wrapMesh, bodyMask, bodyMesh] = await Promise.all([
    api.getWrapMesh(sessionId).catch(() => null),
    fromRefit ? api.getBodyMask(sessionId).catch(() => null) : Promise.resolve(null),
    fromRefit && !opts.targetMesh && r.session.target_ref
      ? api.getMesh(sessionId, 'target_ref').catch(() => null)
      : Promise.resolve(null),
  ])
  const patch: any = {
    wrapping: false,
    jobStatus: null,
    session: r.session,
    wrapMesh,
    resultFromRefit: fromRefit,
    bodyMask,
    refitStats: r.refit ?? null,
    // Jump to the Preview results tab so the user immediately sees the wrap
    // result there (same behavior as a finished /run).
    mode: 'preview',
  }
  if (bodyMesh) patch.targetMesh = bodyMesh
  set(patch)
  savePrefs({ ...loadPrefs(), mode: 'preview' })
  if (opts.download) {
    // Auto-download the .obj so the user can open it in Blender immediately.
    const a = document.createElement('a')
    a.href = api.wrapUrl(sessionId)
    a.rel = 'noopener'
    a.download = '' // let the server-provided Content-Disposition filename win
    document.body.appendChild(a)
    a.click()
    a.remove()
  }
}

export const useStore = create<State>((set, get) => ({
  session: null,
  sourceMesh: null,
  targetMesh: null,
  markers: [],
  pendingPick: null,
  running: false,
  runError: null,
  lastRun: null,

  wrapping: false,
  wrapError: null,
  polishMatrix: null,
  wrapMesh: null,
  resultFromRefit: false,
  bodyMask: null,
  refitStats: null,
  gripPreview: null,
  gripPreviewing: false,
  gripLayer: null,
  // Heatmap on by default: right after a wrap the preview auto-opens and
  // the user should immediately see where the result missed the target.
  showHeatmap: true,

  regionTool: 'off',
  regionWaypoints: [],
  regionClosed: false,
  regionSeed: null,
  regionInvert: false,
  regionLoop: null,
  regionEditable: null,
  regionPreviewInfo: null,
  regionPreviewing: false,
  regionError: null,

  tool: INITIAL_PREFS.tool ?? null,

  queueStatus: null,
  jobStatus: null,

  mode: INITIAL_PREFS.mode ?? 'edit',
  selectedPoseIndex: 0,
  poseMeshes: {},
  resultMeshes: {},
  previewLoading: false,
  previewError: null,
  isPlaying: false,
  fps: INITIAL_PREFS.fps ?? 12,

  lastFbxExtract: null,
  uploadingRef: null,
  uploadError: null,

  symmetryEnabled: INITIAL_PREFS.symmetryEnabled ?? false,
  symmetryAxis: INITIAL_PREFS.symmetryAxis ?? 'X',
  syncRotation: INITIAL_PREFS.syncRotation ?? false,
  showWireframe: INITIAL_PREFS.showWireframe ?? false,

  markerHistory: [],
  markerRedo: [],

  async initSession() {
    // If we already had a session id (from localStorage or in memory), tell
    // the backend to drop it BEFORE we make a new one. This is the user's
    // explicit "start fresh" path — they don't want the old workspace
    // sitting on the server.
    const oldId = get().session?.session_id ?? loadSessionId()
    if (oldId) {
      await api.deleteSession(oldId)
    }
    const s = await api.createSession()
    saveSessionId(s.session_id)
    set({
      session: s,
      sourceMesh: null,
      targetMesh: null,
      markers: [],
      pendingPick: null,
      lastRun: null,
      runError: null,
      mode: 'edit',
      // "New session" means "start over", and starting over begins with the
      // question the tool-selection screen asks — so drop the picked tool and
      // land the user back on it.
      tool: null,
      selectedPoseIndex: 0,
      poseMeshes: {},
      resultMeshes: {},
      isPlaying: false,
      markerHistory: [],
      markerRedo: [],
      wrapMesh: null,
      resultFromRefit: false,
      bodyMask: null,
      refitStats: null,
      wrapError: null,
    })
    // mode/tool reset is the user's expectation when starting fresh; persist
    // it so the next reload matches.
    savePrefs({ ...loadPrefs(), mode: 'edit', tool: null })
  },

  /**
   * On page load, try to pick up the previous session by id from localStorage.
   * Falls back to creating a fresh one if there's nothing saved or the saved
   * session no longer exists (e.g. backend was wiped). Re-fetches meshes +
   * markers so the UI looks exactly like the user left it.
   */
  async restoreOrCreateSession() {
    const savedId = loadSessionId()
    if (savedId) {
      try {
        const existing = await api.fetchSessionIfExists(savedId)
        if (existing) {
          // Got it back. Pull meshes (only the ones that exist) and markers
          // in parallel so the UI lights up at once.
          const [sourceMesh, targetMesh, markers, wrapMesh, bodyMask] = await Promise.all([
            existing.source_ref ? api.getMesh(savedId, 'source_ref').catch(() => null) : Promise.resolve(null),
            existing.target_ref ? api.getMesh(savedId, 'target_ref').catch(() => null) : Promise.resolve(null),
            api.getMarkers(savedId).catch(() => [] as MarkerPair[]),
            existing.has_wrap ? api.getWrapMesh(savedId).catch(() => null) : Promise.resolve(null),
            // Only a refit writes the body mask, so its presence tells us the
            // restored result came from refit → restore the combined preview.
            existing.has_wrap ? api.getBodyMask(savedId).catch(() => null) : Promise.resolve(null),
          ])

          // If there are results already, populate lastRun so the Results
          // panel and Preview tab become available without forcing a re-run.
          const lastRun = existing.has_results
            ? {
                mapping_size: 0,  // unknown after restart; we don't persist this
                results: existing.poses
                  .filter((p) => existing.result_pose_ids.includes(p.pose_id))
                  .map((p) => ({ pose_id: p.pose_id, name: p.name })),
              }
            : null

          set({
            session: existing,
            sourceMesh,
            targetMesh,
            markers,
            pendingPick: null,
            lastRun,
            runError: null,
            poseMeshes: {},
            resultMeshes: {},
            isPlaying: false,
            markerHistory: [],
            markerRedo: [],
            wrapMesh,
            resultFromRefit: !!bodyMask,
            bodyMask,
            refitStats: null,   // stats aren't persisted; only fresh results carry them
            wrapError: null,
          })
          // If a /run or /wrap was still computing when the tab closed, the
          // server kept going — re-attach and resume the progress UI.
          void get().resumeJobIfActive()
          return
        }
        // 404 — saved id is stale, fall through to create.
      } catch {
        // Network blip / backend down — fall through to create.
      }
    }
    await get().initSession()
  },

  async refreshSession() {
    const s = get().session
    if (!s) return
    set({ session: await api.getSession(s.session_id) })
  },

  async uploadReference(role, file) {
    const s = get().session
    if (!s) throw new Error('No session')
    set({ uploadingRef: role, uploadError: null, lastFbxExtract: null })
    try {
      const updated = await api.uploadReference(s.session_id, role, file)
      const mesh = await api.getMesh(s.session_id, role)
      const patch: any = {
        session: updated,
        uploadingRef: null,
        // Backend's invalidate_source/target_cache also drops the wrap result.
        // Mirror that on the client so the stale wrap doesn't keep rendering.
        wrapMesh: null,
        resultFromRefit: false,
        bodyMask: null,
        refitStats: null,
      }
      if (role === 'source_ref') {
        patch.sourceMesh = mesh
        patch.poseMeshes = {}
        patch.resultMeshes = {}
        patch.markers = []
        patch.markerHistory = []
        patch.markerRedo = []
        patch.pendingPick = null
        // Region indices point into the old source topology — reset them.
        patch.regionTool = 'off'
        patch.regionWaypoints = []
        patch.regionClosed = false
        patch.regionSeed = null
        patch.regionInvert = false
        patch.regionLoop = null
        patch.regionEditable = null
        patch.regionPreviewInfo = null
        patch.regionError = null
      } else {
        patch.targetMesh = mesh
      }
      if (updated.fbx_extract) {
        patch.lastFbxExtract = updated.fbx_extract
      }
      set(patch)
    } catch (e: any) {
      set({ uploadingRef: null, uploadError: e?.message ?? String(e) })
      throw e
    }
  },

  dismissFbxExtract() {
    set({ lastFbxExtract: null })
  },

  dismissUploadError() {
    set({ uploadError: null })
  },

  async uploadPoses(files) {
    const s = get().session
    if (!s) throw new Error('No session')
    const r = await api.uploadPoses(s.session_id, files)
    set({ session: r.session })
    return { rejected: r.rejected }
  },

  async deletePose(poseId) {
    const s = get().session
    if (!s) return
    const updated = await api.deletePose(s.session_id, poseId)
    const { poseMeshes, resultMeshes } = get()
    const nextPose = { ...poseMeshes }
    const nextRes = { ...resultMeshes }
    delete nextPose[poseId]
    delete nextRes[poseId]
    set({ session: updated, poseMeshes: nextPose, resultMeshes: nextRes })
  },

  pickVertex(side, vertexIdx) {
    const st = get()
    const { pendingPick, markers, symmetryEnabled, symmetryAxis, sourceMesh, targetMesh } = st

    // No pending pick yet — start one on whichever side the user clicked.
    if (pendingPick === null) {
      set({ pendingPick: { side, vertexIdx } })
      return
    }

    // Same side as pending — toggle off if same vertex, replace otherwise.
    if (pendingPick.side === side) {
      set({ pendingPick: pendingPick.vertexIdx === vertexIdx ? null : { side, vertexIdx } })
      return
    }

    // Different side — finalize a (source, target) pair.
    const sourceIdx = pendingPick.side === 'source' ? pendingPick.vertexIdx : vertexIdx
    const targetIdx = pendingPick.side === 'source' ? vertexIdx : pendingPick.vertexIdx

    // Duplicate guard
    if (markers.some((m) => m.source === sourceIdx && m.target === targetIdx)) {
      set({ pendingPick: null })
      return
    }

    const pairsToAdd: MarkerPair[] = [{ source: sourceIdx, target: targetIdx }]

    // --- Optional symmetric companion pair ---
    if (symmetryEnabled && sourceMesh && targetMesh) {
      const ax = axisToIndex(symmetryAxis)
      const sourceR = meshBoundingRadius(sourceMesh.vertices)
      const targetR = meshBoundingRadius(targetMesh.vertices)
      const sTol = defaultMirrorTolerance(sourceR)
      const tTol = defaultMirrorTolerance(targetR)
      const sourceAnchor = meshCentroid(sourceMesh.vertices)
      const targetAnchor = meshCentroid(targetMesh.vertices)
      const mirrorSource = findMirrorVertex(sourceMesh.vertices, sourceIdx, ax, sTol, sourceAnchor)
      const mirrorTarget = findMirrorVertex(targetMesh.vertices, targetIdx, ax, tTol, targetAnchor)
      if (
        mirrorSource !== null && mirrorSource !== sourceIdx &&
        mirrorTarget !== null && mirrorTarget !== targetIdx &&
        !markers.some((m) => m.source === mirrorSource && m.target === mirrorTarget) &&
        !pairsToAdd.some((m) => m.source === mirrorSource && m.target === mirrorTarget)
      ) {
        pairsToAdd.push({ source: mirrorSource, target: mirrorTarget })
      } else {
        // Surface why nothing was added so the user can debug from devtools.
        // Common reasons: wrong axis, mesh not actually symmetric, or the
        // picked vertex is right on the mirror plane.
        const why = (label: string, mirror: number | null, picked: number, tol: number, anchor: number[]) => {
          if (mirror === null) return `${label}: no vertex within tol=${tol.toExponential(2)} of mirror around ${anchor.map(v => v.toFixed(3))}`
          if (mirror === picked) return `${label}: closest match is the picked vertex itself (sits on mirror plane)`
          return `${label}: mirror=${mirror} (might be a duplicate of an existing marker)`
        }
        // eslint-disable-next-line no-console
        console.warn(
          `[symmetry] no companion pair added (axis=${symmetryAxis}). ` +
          why('source', mirrorSource, sourceIdx, sTol, sourceAnchor) + '; ' +
          why('target', mirrorTarget, targetIdx, tTol, targetAnchor)
        )
      }
    }

    const next = [...markers, ...pairsToAdd]
    set({ ...pushHistory(st), markers: next, pendingPick: null })
    get().syncMarkers()
  },

  addMarkerPair(source, target) {
    const st = get()
    if (st.markers.some((m) => m.source === source && m.target === target)) return
    const next = [...st.markers, { source, target }]
    set({ ...pushHistory(st), markers: next, pendingPick: null })
    get().syncMarkers()
  },

  removeMarker(idx) {
    const st = get()
    const { markers, symmetryEnabled, symmetryAxis, sourceMesh, targetMesh } = st
    const victim = markers[idx]
    if (!victim) return

    // Indices to drop. With symmetric editing on, a pair is created together
    // (see pickVertex) — so it should be destroyed together: also remove the
    // mirror companion if one exists.
    const drop = new Set<number>([idx])
    if (symmetryEnabled && sourceMesh && targetMesh) {
      const ax = axisToIndex(symmetryAxis)
      const sTol = defaultMirrorTolerance(meshBoundingRadius(sourceMesh.vertices))
      const tTol = defaultMirrorTolerance(meshBoundingRadius(targetMesh.vertices))
      const sAnchor = meshCentroid(sourceMesh.vertices)
      const tAnchor = meshCentroid(targetMesh.vertices)
      const ms = findMirrorVertex(sourceMesh.vertices, victim.source, ax, sTol, sAnchor)
      const mt = findMirrorVertex(targetMesh.vertices, victim.target, ax, tTol, tAnchor)
      // Skip self-mirror (vertex on the symmetry plane) — there is no distinct
      // companion to remove in that case.
      if (ms !== null && mt !== null && (ms !== victim.source || mt !== victim.target)) {
        const companion = markers.findIndex(
          (m, i) => i !== idx && m.source === ms && m.target === mt,
        )
        if (companion !== -1) drop.add(companion)
      }
    }

    const next = markers.filter((_, i) => !drop.has(i))
    set({ ...pushHistory(st), markers: next })
    get().syncMarkers()
  },

  clearMarkers() {
    const st = get()
    if (st.markers.length === 0) return
    set({ ...pushHistory(st), markers: [], pendingPick: null })
    get().syncMarkers()
  },

  resetPendingPick() {
    set({ pendingPick: null })
  },

  async syncMarkers() {
    const s = get().session
    if (!s) return
    await api.putMarkers(s.session_id, get().markers)
  },

  exportMarkers() {
    const st = get()
    const blob = new Blob(
      [
        JSON.stringify(
          {
            version: 1,
            tool: 'chameleon-warp',
            exported_at: new Date().toISOString(),
            source_ref: st.session?.source_ref?.name ?? null,
            target_ref: st.session?.target_ref?.name ?? null,
            source_vertex_count: st.sourceMesh?.vertex_count ?? null,
            target_vertex_count: st.targetMesh?.vertex_count ?? null,
            markers: st.markers,
          },
          null,
          2,
        ),
      ],
      { type: 'application/json' },
    )
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    a.download = `markers-${stamp}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Revoke on next tick so the click has time to fire in Safari/Firefox.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  },

  async importMarkers(file) {
    const st = get()
    const text = await file.text()
    let data: any
    try {
      data = JSON.parse(text)
    } catch (e: any) {
      throw new Error(`Invalid JSON: ${e?.message ?? e}`)
    }

    // Accept two shapes:
    //   1. our own export: { version, markers: [{source, target}, ...] }
    //   2. bare array:     [{source, target}, ...]
    const rawList: any = Array.isArray(data) ? data : data?.markers
    if (!Array.isArray(rawList)) {
      throw new Error('No "markers" array found in file')
    }

    const sourceVerts = st.sourceMesh?.vertex_count ?? 0
    const targetVerts = st.targetMesh?.vertex_count ?? 0

    const existingSet = new Set(st.markers.map((m) => `${m.source}|${m.target}`))

    const accepted: MarkerPair[] = []
    let skipped = 0   // duplicate or already present
    let invalid = 0   // bad shape or out of range

    for (const entry of rawList) {
      const s = Number(entry?.source)
      const t = Number(entry?.target)
      if (!Number.isInteger(s) || !Number.isInteger(t) || s < 0 || t < 0) {
        invalid++
        continue
      }
      if (sourceVerts && s >= sourceVerts) { invalid++; continue }
      if (targetVerts && t >= targetVerts) { invalid++; continue }
      const key = `${s}|${t}`
      if (existingSet.has(key)) { skipped++; continue }
      existingSet.add(key)
      accepted.push({ source: s, target: t })
    }

    if (accepted.length === 0 && invalid === 0 && skipped > 0) {
      // Everything was already present — nothing to do, but tell the caller.
      return { added: 0, skipped, invalid, warning: 'All markers in file were already present.' }
    }

    if (accepted.length === 0 && invalid > 0) {
      throw new Error(
        `None of the ${rawList.length} markers in the file are valid for the current ` +
        `references (source has ${sourceVerts} verts, target has ${targetVerts}).`,
      )
    }

    // Build optional mismatch warning so user can spot a wrong-file mistake.
    let warning: string | undefined
    if (data && !Array.isArray(data)) {
      const fileSourceName = data.source_ref ?? null
      const fileTargetName = data.target_ref ?? null
      const ourSourceName = st.session?.source_ref?.name ?? null
      const ourTargetName = st.session?.target_ref?.name ?? null
      if (fileSourceName && ourSourceName && fileSourceName !== ourSourceName) {
        warning = `File was exported with source "${fileSourceName}" but current source is "${ourSourceName}".`
      } else if (fileTargetName && ourTargetName && fileTargetName !== ourTargetName) {
        warning = `File was exported with target "${fileTargetName}" but current target is "${ourTargetName}".`
      }
    }

    const next = [...st.markers, ...accepted]
    set({ ...pushHistory(st), markers: next, pendingPick: null })
    await get().syncMarkers()
    return { added: accepted.length, skipped, invalid, warning }
  },

  async run(opts) {
    const s = get().session
    if (!s) return
    const sid = s.session_id
    set({ running: true, runError: null, jobStatus: null })
    try {
      // Submit the job. A 409 means one is already active for this session
      // (e.g. a double click, or a resume racing us) — just attach to it.
      try {
        await api.submitRun(sid, opts)
      } catch (e: any) {
        if (!String(e?.message ?? e).includes('409')) throw e
      }
      const job = await pollJob(sid, set, () => get().session?.session_id === sid)
      if (!job) {
        set({
          running: false,
          jobStatus: null,
          runError: 'Lost track of the job — it may still be running on the server. Reload to reattach.',
        })
        return
      }
      if (job.state === 'error') {
        set({ running: false, jobStatus: null, runError: job.error ?? 'Job failed' })
        return
      }
      applyRunDone(job.result as RunResponse, set, get)
    } catch (e: any) {
      set({ running: false, jobStatus: null, runError: e?.message ?? String(e) })
    }
  },

  async wrap(opts) {
    const s = get().session
    if (!s) return
    const sid = s.session_id
    set({ wrapping: true, wrapError: null, jobStatus: null })
    try {
      try {
        await api.submitWrap(sid, opts)
      } catch (e: any) {
        if (!String(e?.message ?? e).includes('409')) throw e
      }
      const job = await pollJob(sid, set, () => get().session?.session_id === sid)
      if (!job) {
        set({
          wrapping: false,
          jobStatus: null,
          wrapError: 'Lost track of the job — it may still be running on the server. Reload to reattach.',
        })
        return
      }
      if (job.state === 'error') {
        set({ wrapping: false, jobStatus: null, wrapError: job.error ?? 'Job failed' })
        return
      }
      await applyWrapDone(sid, job.result as WrapResponse, set,
        { download: true, targetMesh: get().targetMesh })
    } catch (e: any) {
      set({ wrapping: false, jobStatus: null, wrapError: e?.message ?? String(e) })
    }
  },

  async refit(opts) {
    // Refit saves its result into the same 'wrap' slot, so it reuses the wrap
    // progress/error state and applyWrapDone (preview + download) unchanged.
    const s = get().session
    if (!s) return
    const sid = s.session_id
    set({ wrapping: true, wrapError: null, jobStatus: null })
    try {
      try {
        await api.submitRefit(sid, opts)
      } catch (e: any) {
        if (!String(e?.message ?? e).includes('409')) throw e
      }
      const job = await pollJob(sid, set, () => get().session?.session_id === sid)
      if (!job) {
        set({
          wrapping: false,
          jobStatus: null,
          wrapError: 'Lost track of the job — it may still be running on the server. Reload to reattach.',
        })
        return
      }
      if (job.state === 'error') {
        set({ wrapping: false, jobStatus: null, wrapError: job.error ?? 'Job failed' })
        return
      }
      const result = job.result as WrapResponse
      // Auto-polish: adopt the effective placement into the gizmo (App.tsx).
      const pm = result.refit?.polish?.matrix
      if (Array.isArray(pm) && pm.length === 16) set({ polishMatrix: pm })
      await applyWrapDone(sid, result, set,
        { download: true, targetMesh: get().targetMesh })
    } catch (e: any) {
      set({ wrapping: false, jobStatus: null, wrapError: e?.message ?? String(e) })
    }
  },

  async refitViaProxy(opts) {
    // Same job/preview plumbing as refit — the result lands in the wrap slot.
    const s = get().session
    if (!s) return
    const sid = s.session_id
    set({ wrapping: true, wrapError: null, jobStatus: null })
    try {
      try {
        await api.refitViaProxy(sid, opts)
      } catch (e: any) {
        if (!String(e?.message ?? e).includes('409')) throw e
      }
      const job = await pollJob(sid, set, () => get().session?.session_id === sid)
      if (!job) {
        set({
          wrapping: false,
          jobStatus: null,
          wrapError: 'Lost track of the job — it may still be running on the server. Reload to reattach.',
        })
        return
      }
      if (job.state === 'error') {
        set({ wrapping: false, jobStatus: null, wrapError: job.error ?? 'Job failed' })
        return
      }
      await applyWrapDone(sid, job.result as WrapResponse, set,
        { download: true, targetMesh: get().targetMesh })
    } catch (e: any) {
      set({ wrapping: false, jobStatus: null, wrapError: e?.message ?? String(e) })
    }
  },

  async resumeJobIfActive() {
    const s = get().session
    if (!s) return
    const sid = s.session_id
    const job = await api.getJob(sid).catch(() => null)
    if (!job || (job.state !== 'queued' && job.state !== 'running')) return
    // A compute is still going server-side — reflect it and poll to completion.
    if (job.kind === 'run') set({ running: true, runError: null, jobStatus: job })
    else set({ wrapping: true, wrapError: null, jobStatus: job })
    const done = await pollJob(sid, set, () => get().session?.session_id === sid)
    if (!done) {
      set(job.kind === 'run' ? { running: false, jobStatus: null } : { wrapping: false, jobStatus: null })
      return
    }
    if (done.state === 'error') {
      set(
        job.kind === 'run'
          ? { running: false, jobStatus: null, runError: done.error ?? 'Job failed' }
          : { wrapping: false, jobStatus: null, wrapError: done.error ?? 'Job failed' },
      )
      return
    }
    if (done.kind === 'run') applyRunDone(done.result as RunResponse, set, get)
    else await applyWrapDone(sid, done.result as WrapResponse, set,
      { download: false, targetMesh: get().targetMesh })
  },

  setShowHeatmap(v) {
    set({ showHeatmap: v })
  },

  dismissWrapError() {
    set({ wrapError: null })
  },

  async saveBodyMask(hidden) {
    const s = get().session
    if (!s) return null
    const mask = await api.putBodyMask(s.session_id, hidden)
    set({ bodyMask: mask })
    return mask
  },

  async dilateBodyMask(rings, hidden) {
    const s = get().session
    if (!s) return null
    // Persist the on-screen edits first so the grow/shrink acts on them, then
    // dilate the persisted mask.
    await api.putBodyMask(s.session_id, hidden)
    const mask = await api.dilateBodyMask(s.session_id, rings)
    set({ bodyMask: mask })
    return mask
  },

  async previewGrip(opts) {
    const s = get().session
    if (!s) return
    set({ gripPreviewing: true })
    try {
      const r = await api.refitField(s.session_id, opts)
      set({ gripPreview: r.grip_frac, gripLayer: r.layer ?? null, gripPreviewing: false })
    } catch {
      set({ gripPreview: null, gripLayer: null, gripPreviewing: false })
    }
  },

  clearGripPreview() {
    if (get().gripPreview !== null || get().gripLayer !== null || get().gripPreviewing)
      set({ gripPreview: null, gripLayer: null, gripPreviewing: false })
  },

  // ---- region drawing ----
  setRegionTool(t) {
    set({ regionTool: t })
  },

  addRegionWaypoint(v) {
    const { regionWaypoints: wp, regionClosed } = get()
    // A closed loop is locked — undo/clear to edit it again.
    if (regionClosed) return
    // Clicking the start point (with >=3 points down) closes the loop instead
    // of adding a duplicate vertex. This is the explicit "click start to close"
    // gesture — the backend stitches the final start<->end segment itself.
    if (wp.length >= 3 && v === wp[0]) {
      set({ regionClosed: true, regionLoop: null, regionEditable: null, regionPreviewInfo: null })
      return
    }
    // Ignore an immediate repeat-click on the same vertex.
    if (wp.length && wp[wp.length - 1] === v) return
    // Editing the boundary invalidates any stale preview overlay.
    set({
      regionWaypoints: [...wp, v],
      regionLoop: null, regionEditable: null, regionPreviewInfo: null,
    })
  },

  closeRegionLoop() {
    const { regionWaypoints: wp } = get()
    if (wp.length < 3) {
      set({ regionError: 'Click at least 3 boundary points before closing' })
      return
    }
    set({ regionClosed: true, regionLoop: null, regionEditable: null, regionPreviewInfo: null })
  },

  undoRegionWaypoint() {
    // If the loop is closed, undo reopens it (keeps the points). Otherwise
    // it removes the last point.
    if (get().regionClosed) {
      set({ regionClosed: false, regionLoop: null, regionEditable: null, regionPreviewInfo: null })
      return
    }
    set({
      regionWaypoints: get().regionWaypoints.slice(0, -1),
      regionLoop: null, regionEditable: null, regionPreviewInfo: null,
    })
  },

  clearRegion() {
    set({
      regionWaypoints: [], regionClosed: false, regionSeed: null, regionInvert: false,
      regionLoop: null, regionEditable: null, regionPreviewInfo: null,
      regionError: null,
    })
  },

  setRegionSeed(v) {
    set({ regionSeed: v, regionEditable: null, regionPreviewInfo: null })
  },

  setRegionInvert(v) {
    // Flipping the side invalidates the current preview overlay.
    set({ regionInvert: v, regionLoop: null, regionEditable: null, regionPreviewInfo: null })
  },

  async previewRegion(featherWidth, featherExponent) {
    const s = get().session
    const { regionWaypoints, regionSeed, regionInvert, regionClosed } = get()
    if (!s) return
    if (regionWaypoints.length < 3) {
      set({ regionError: 'Click at least 3 boundary points to close a loop' })
      return
    }
    if (!regionClosed) {
      set({ regionError: 'Close the loop first (click the start point or “Close loop”)' })
      return
    }
    set({ regionPreviewing: true, regionError: null })
    try {
      const pv = await api.previewRegion(s.session_id, {
        waypoints: regionWaypoints, seed: regionSeed, invert: regionInvert,
        feather_width: featherWidth, feather_exponent: featherExponent,
      })
      set({
        regionPreviewing: false,
        regionLoop: pv.loop, regionEditable: pv.interior, regionPreviewInfo: pv,
      })
    } catch (e: any) {
      set({
        regionPreviewing: false, regionError: e?.message ?? String(e),
        regionLoop: null, regionEditable: null, regionPreviewInfo: null,
      })
    }
  },

  dismissRegionError() {
    set({ regionError: null })
  },

  setTool(tool) {
    set({ tool })
    savePrefs({ ...loadPrefs(), tool })
  },

  setMode(mode) {
    set({ mode, isPlaying: false })
    savePrefs({ ...loadPrefs(), mode })
    if (mode === 'preview') get().loadPreviewMeshes()
  },

  setSelectedPoseIndex(idx) {
    set({ selectedPoseIndex: idx })
  },

  async loadPreviewMeshes(priorityPoseId) {
    const s = get().session
    if (!s) return
    // Current frame first: the viewport becomes usable after ONE pose+result
    // pair instead of after the whole set (with 50 poses that's minutes).
    const sorted = sortPoses(s.poses)
    if (priorityPoseId) {
      const i = sorted.findIndex((p) => p.pose_id === priorityPoseId)
      if (i > 0) sorted.unshift(...sorted.splice(i, 1))
    }

    // Meshes commit to the store AS THEY ARRIVE (not in one set at the end),
    // and every fetch registers in a shared in-flight map — so the multiple
    // callers of this action (run(), setMode('preview'), PreviewPanel mount
    // and frame changes, StrictMode double-effects) share one HTTP request
    // per mesh instead of each re-downloading the entire set.
    type Task = { key: string; run: () => Promise<void> }
    const alive = () => get().session?.session_id === s.session_id
    const tasks: Task[] = []
    for (const p of sorted) {
      if (!get().poseMeshes[p.pose_id]) {
        tasks.push({
          key: `pose:${s.session_id}:${p.pose_id}`,
          run: async () => {
            const mesh = await api.getPoseMesh(s.session_id, p.pose_id)
            if (alive()) set({ poseMeshes: { ...get().poseMeshes, [p.pose_id]: mesh } })
          },
        })
      }
      if (s.result_pose_ids.includes(p.pose_id) && !get().resultMeshes[p.pose_id]) {
        tasks.push({
          key: `res:${s.session_id}:${p.pose_id}`,
          run: async () => {
            const mesh = await api.getResultMesh(s.session_id, p.pose_id)
            if (alive()) set({ resultMeshes: { ...get().resultMeshes, [p.pose_id]: mesh } })
          },
        })
      }
    }
    if (tasks.length === 0) return

    set({ previewLoading: true, previewError: null })
    try {
      const runShared = (t: Task): Promise<void> => {
        let p = meshFetchInflight.get(t.key)
        if (!p) {
          p = t.run().finally(() => meshFetchInflight.delete(t.key))
          meshFetchInflight.set(t.key, p)
        }
        return p
      }
      const BATCH = 6
      for (let i = 0; i < tasks.length; i += BATCH) {
        await Promise.all(tasks.slice(i, i + BATCH).map(runShared))
      }
      set({ previewLoading: false })
    } catch (e: any) {
      set({ previewLoading: false, previewError: e?.message ?? String(e) })
    }
  },

  togglePlay() {
    set({ isPlaying: !get().isPlaying })
  },

  setFps(fps) {
    const clamped = Math.max(1, Math.min(60, fps))
    set({ fps: clamped })
    savePrefs({ ...loadPrefs(), fps: clamped })
  },

  setSymmetryEnabled(v) {
    set({ symmetryEnabled: v })
    savePrefs({ ...loadPrefs(), symmetryEnabled: v })
  },

  setSymmetryAxis(a) {
    set({ symmetryAxis: a })
    savePrefs({ ...loadPrefs(), symmetryAxis: a })
  },

  setSyncRotation(v) {
    set({ syncRotation: v })
    savePrefs({ ...loadPrefs(), syncRotation: v })
  },

  setShowWireframe(v) {
    set({ showWireframe: v })
    savePrefs({ ...loadPrefs(), showWireframe: v })
  },

  undo() {
    const st = get()
    if (st.markerHistory.length === 0) return
    const prev = st.markerHistory[st.markerHistory.length - 1]
    const nextHistory = st.markerHistory.slice(0, -1)
    const nextRedo = [...st.markerRedo, [...st.markers]]
    set({
      markerHistory: nextHistory,
      markerRedo: nextRedo,
      markers: prev,
      pendingPick: null,
    })
    get().syncMarkers()
  },

  redo() {
    const st = get()
    if (st.markerRedo.length === 0) return
    const next = st.markerRedo[st.markerRedo.length - 1]
    const nextRedo = st.markerRedo.slice(0, -1)
    const nextHistory = [...st.markerHistory, [...st.markers]]
    set({
      markerRedo: nextRedo,
      markerHistory: nextHistory,
      markers: next,
      pendingPick: null,
    })
    get().syncMarkers()
  },

  canUndo() {
    return get().markerHistory.length > 0
  },

  canRedo() {
    return get().markerRedo.length > 0
  },
}))
