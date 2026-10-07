export type Role = 'source_ref' | 'target_ref' | 'proxy_ref'

export interface SessionSummary {
  session_id: string
  source_ref: { name: string } | null
  target_ref: { name: string } | null
  proxy_ref?: { name: string } | null
  source_uploaded_as_fbx: boolean
  target_uploaded_as_fbx: boolean
  // False when the uploaded target is too dense for DT transfer (Run) but
  // still fine for Wrap.
  target_dt_available?: boolean
  poses: { pose_id: string; name: string }[]
  marker_count: number
  result_pose_ids: string[]
  has_results: boolean
  has_wrap?: boolean
  // Persisted refit working state (item 3a) — opaque blob, see RefitState.
  refit_state?: RefitState | null
  // Session lifetime (Unix seconds). expires_at = last_seen + ttl_seconds.
  last_seen?: number
  ttl_seconds?: number
  expires_at?: number
}

// Refit working state persisted with the session (item 3a). Captured against a
// source name so a changed garment discards it. pins mirror App's RefitPinEntry.
export interface RefitState {
  source_name: string | null
  pre_transform: number[] | null
  pins: { marker: { source: number; target: number } | null; preserve: number[] }[]
  frozen: number[]
  preset: string
  advanced: { contact_tight: string; contact_free: string; thickness: string }
}

export interface FbxExtractInfo {
  mesh_name: string
  vertex_count: number
  polygon_count: number
  extracted_poses: string[]
}

export interface UploadReferenceResponse extends SessionSummary {
  fbx_extract?: FbxExtractInfo
  // Present when a target uploaded fine but is too dense for DT transfer.
  target_warning?: string
}

export interface ResidualStats {
  mean: number
  p95: number
  max: number
  bbox_diag: number
}

export interface MeshData {
  vertices: number[]      // flat (x,y,z,...) length = vertex_count*3
  faces: number[]         // flat (a,b,c,...) length = face_count*3
  vertex_count: number
  face_count: number
  // Wrap result only: per-vertex distance to the target surface as a
  // fraction of its bbox diagonal (-1 = frozen vertex, no data), plus the
  // aggregate stats. Feeds the error heatmap + the numbers in the panel.
  distances_frac?: number[]
  residual?: ResidualStats
  // Refit only: the conform ("grip") field the auto-detection built, 0..1
  // per vertex (1 = pulled onto the body, 0 = left free / released layer).
  grip_frac?: number[]
}

export interface MarkerPair {
  source: number
  target: number
}

// The covered-body-vertex mask for a refit result: which TARGET (body) verts
// sit under the fitted garment, so the preview can hide that skin.
export interface BodyMask {
  hidden_vertices: number[]
  body_vertex_count: number
}

export interface RunResponse {
  session: SessionSummary
  mapping_size: number
  // True when source/target topologies were identical: correspondence was
  // skipped and the mapping is the per-triangle identity (markers unused).
  identity_mapping?: boolean
  results: { pose_id: string; name: string }[]
  // True when the results restored the target's original quads/n-gons + UVs.
  polygons_preserved?: boolean
}

// --- async job model ---------------------------------------------------------
// /run and /wrap return a job_id immediately; the browser polls GET /job for
// state + progress + queue position until the compute finishes.
export type JobState = 'queued' | 'running' | 'done' | 'error'

export interface JobProgress {
  stage: string   // queued|precompute|iterations|transfer|projection|export|done|error
  iter: number
  total: number   // 0 = indeterminate (no bar)
}

export interface JobSubmit {
  job_id: string
  kind: 'run' | 'wrap'
  state: JobState
}

export interface JobStatus {
  job_id: string
  kind: 'run' | 'wrap'
  state: JobState
  queue_position: number   // 0 = running/next, N = N jobs ahead, -1 = terminal
  progress: JobProgress
  error: string | null
  // Present only when state === 'done': the endpoint's result payload
  // (a RunResponse or WrapResponse depending on kind).
  result?: RunResponse | WrapResponse
}

// Refit result diagnostics (the spread of RefitResult.stats). Drives the stats
// block + warnings in the result panel. All optional — older results / restores
// may omit them. polish.matrix (16 floats) is the effective placement when
// auto-polish fired.
export interface RefitStats {
  preset?: string
  collision?: string
  contact_verts?: number
  occluded_verts?: number
  opposed_verts?: number
  matched_verts?: number
  frozen_verts?: number
  preserve_verts?: number
  seam_grip_verts?: number
  rigidity_dropped?: number
  hidden_body_verts?: number
  bind?: { free?: number; handles?: number; iterations?: number }
  polish?: { matrix?: number[] } | null
  [k: string]: unknown
}

export interface WrapResponse {
  session: SessionSummary
  vertex_count: number
  face_count: number
  elapsed_seconds: number
  align_scale?: number | null
  residual?: ResidualStats | null
  polygons_preserved?: boolean
  refit?: RefitStats
}

// Full option set for a /wrap submission. The advanced surface-matching knobs
// map 1:1 to WrapPayload fields on the backend (dt_core.surface).
export interface WrapOpts {
  iterations: number
  smoothness: number
  identity_weight: number
  use_closest_point?: boolean
  region?: RegionParams
  align_to_source?: boolean
  smooth_result?: number
  // --- advanced (optional; backend has sane defaults) ---
  match_max_angle_deg?: number   // normal-compatibility limit for matches
  match_distance_frac?: number   // match-distance floor, fraction of bbox diag
  project_result?: boolean       // final snap onto target surface
  project_distance_frac?: number // max snap distance, fraction of bbox diag
  soft_markers?: boolean         // relax marker pins in the strong phase
}

// Refit: fit a garment/accessory (source) onto a body (target). A preset picks
// the conform field + stiffness + collision; advanced fields override it.
export type RefitPreset = 'accessory' | 'cloth' | 'armor' | 'skintight'
export interface RefitOpts {
  preset: RefitPreset
  iterations?: number
  pre_transform?: number[] | null  // row-major 4x4 (16 floats); gizmo place/resize
  grip_width?: number | null
  offset?: number | null
  stiffness?: number | null
  smoothness?: number | null
  collision?: 'none' | 'push_out' | 'hide_body' | null
  // Pin "preserve" (outer-wall) source verts to release; inner grip is a marker.
  preserve?: number[]
  pin_radius?: number | null
  // Frozen paint: source verts that must not deform at all — no grip, ride the
  // fit as a near-rigid patch, untouched by collision.
  frozen?: number[]
  // Contact band / cloth thickness overrides (world units); null = preset.
  contact_tight?: number | null
  contact_free?: number | null
  thickness?: number | null
  // Auto-polish: small trust-regioned pre-alignment of the placement. When it
  // fires, the effective placement rides back in stats.refit.polish.matrix.
  auto_polish?: boolean
  // Layer classification: 'legacy' (per-vertex occlusion|opposed) or 'auto'
  // (score-based body-side voting + fan-out occlusion, smoothed + component
  // policy). Auto adds a per-vertex `layer` id to the refit_field response.
  layer_mode?: 'legacy' | 'auto'
}

// A boundary-loop region for partial (masked) wrap.
export interface RegionParams {
  waypoints: number[]
  seed?: number | null   // omit to auto-pick the smaller side
  invert?: boolean       // flip to the other side
  feather_width: number
  // Seam sharpness: how steeply the feather falls off from the boundary
  // inward. 1 = gentle/linear, higher = sharper seam. Backend default 2.
  feather_exponent?: number
}

export interface RegionPreview {
  loop: number[]          // dense stitched boundary loop (source vertex indices)
  interior: number[]      // editable interior vertex indices
  interior_count: number
  frozen_count: number
  feather_count: number
  total: number
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return (await res.json()) as T
}

export const api = {
  async createSession(): Promise<SessionSummary> {
    return json(await fetch('/api/sessions', { method: 'POST' }))
  },

  async getSession(id: string): Promise<SessionSummary> {
    return json(await fetch(`/api/sessions/${id}`))
  },

  async deleteSession(id: string): Promise<void> {
    // Fire-and-forget: we don't care if the session was already gone.
    await fetch(`/api/sessions/${id}`, { method: 'DELETE' }).catch(() => {})
  },

  /** Build a tab-close beacon body for `navigator.sendBeacon`. */
  deleteSessionBeacon(id: string): boolean {
    if (typeof navigator === 'undefined' || !navigator.sendBeacon) return false
    // sendBeacon doesn't support DELETE method, so we POST to a special
    // path that the backend interprets as "tombstone this session".
    // Simpler: just use fetch with keepalive:true which DOES support DELETE
    // and survives the unload event in modern browsers.
    try {
      fetch(`/api/sessions/${id}`, { method: 'DELETE', keepalive: true }).catch(() => {})
      return true
    } catch {
      return false
    }
  },

  async uploadReference(id: string, role: Role, file: File): Promise<UploadReferenceResponse> {
    const fd = new FormData()
    fd.append('file', file)
    return json(await fetch(`/api/sessions/${id}/upload/${role}`, { method: 'POST', body: fd }))
  },

  async uploadPoses(id: string, files: File[]): Promise<{
    session: SessionSummary
    accepted: { pose_id: string; name: string }[]
    rejected: { name: string; reason: string }[]
  }> {
    const fd = new FormData()
    for (const f of files) fd.append('files', f)
    return json(await fetch(`/api/sessions/${id}/upload/poses`, { method: 'POST', body: fd }))
  },

  async deletePose(id: string, poseId: string): Promise<SessionSummary> {
    return json(await fetch(`/api/sessions/${id}/poses/${poseId}`, { method: 'DELETE' }))
  },

  async getMesh(id: string, role: Role): Promise<MeshData> {
    return json(await fetch(`/api/sessions/${id}/mesh/${role}`))
  },

  async getPoseMesh(id: string, poseId: string): Promise<MeshData> {
    return json(await fetch(`/api/sessions/${id}/poses/${poseId}/mesh`))
  },

  async getResultMesh(id: string, poseId: string): Promise<MeshData> {
    return json(await fetch(`/api/sessions/${id}/results/${poseId}/mesh`))
  },

  async putMarkers(id: string, markers: MarkerPair[]): Promise<{ marker_count: number }> {
    return json(
      await fetch(`/api/sessions/${id}/markers`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ markers }),
      }),
    )
  },

  async getMarkers(id: string): Promise<MarkerPair[]> {
    const r = await json<{ markers: MarkerPair[] }>(await fetch(`/api/sessions/${id}/markers`))
    return r.markers
  },

  /** Fetch a session by id WITHOUT throwing on 404. Returns null if not found. */
  async fetchSessionIfExists(id: string): Promise<SessionSummary | null> {
    const res = await fetch(`/api/sessions/${id}`)
    if (res.status === 404) return null
    if (!res.ok) {
      const text = await res.text()
      throw new Error(`${res.status} ${res.statusText}: ${text}`)
    }
    return (await res.json()) as SessionSummary
  },

  /** Submit a /run job. Returns immediately with a job_id; poll getJob(). */
  async submitRun(id: string, opts: { iterations: number; smoothness: number; identity_weight: number }): Promise<JobSubmit> {
    return json(
      await fetch(`/api/sessions/${id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      }),
    )
  },

  /** Submit a /wrap job. Returns immediately with a job_id; poll getJob(). */
  async submitWrap(
    id: string,
    opts: WrapOpts,
  ): Promise<JobSubmit> {
    return json(
      await fetch(`/api/sessions/${id}/wrap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      }),
    )
  },

  /** Dry-run "preview grip": build only the conform field (no solve) and get
   *  it back to color the source mesh. Synchronous on the backend. */
  async refitField(id: string, opts: RefitOpts): Promise<{
    grip_frac: number[]
    // Auto layer mode only: per-vertex layer id (0 driven / 1 follower-wall /
    // 2 follower-component). Absent in legacy mode.
    layer?: number[]
    stats: {
      contact_verts: number; occluded_verts: number; opposed_verts: number
      seam_grip_verts: number
      layers?: {
        mode: string; n_active?: number; driven?: number
        follower_wall?: number; follower_component?: number
        coverage_frac?: number; e_cls?: number
        components?: number; demoted_components?: number
      }
    }
  }> {
    return json(
      await fetch(`/api/sessions/${id}/refit_field`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      }),
    )
  },

  /** Submit a /refit job (garment/accessory onto body). Poll getJob(). */
  async submitRefit(id: string, opts: RefitOpts): Promise<JobSubmit> {
    return json(
      await fetch(`/api/sessions/${id}/refit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      }),
    )
  },

  /** Wear a garment via the proxy basemesh: transport it through the current
   *  wrap result, then a short cleanup refit. Requires a wrap result + an
   *  uploaded proxy_ref basemesh. Placement comes from the binding. */
  async refitViaProxy(id: string, opts: RefitOpts): Promise<JobSubmit> {
    return json(
      await fetch(`/api/sessions/${id}/refit_via_proxy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      }),
    )
  },

  /** Current (or last) background job for the session. null on 404 (no job). */
  async getJob(id: string): Promise<JobStatus | null> {
    const res = await fetch(`/api/sessions/${id}/job`)
    if (res.status === 404) return null
    return json(res)
  },

  async previewRegion(id: string, params: RegionParams): Promise<RegionPreview> {
    return json(
      await fetch(`/api/sessions/${id}/region/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      }),
    )
  },

  async getWrapMesh(id: string): Promise<MeshData> {
    return json(await fetch(`/api/sessions/${id}/mesh/wrap`))
  },

  /** Covered-body mask for the current refit result. null on 404 (no refit,
   *  or a non-refit wrap result). */
  async getBodyMask(id: string): Promise<BodyMask | null> {
    const res = await fetch(`/api/sessions/${id}/body_mask`)
    if (res.status === 404) return null
    return json(res)
  },

  /** Replace the covered-body mask (persist paint edits). */
  async putBodyMask(id: string, hidden: number[]): Promise<BodyMask> {
    return json(
      await fetch(`/api/sessions/${id}/body_mask`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden_vertices: hidden }),
      }),
    )
  },

  /** Grow (rings > 0) / shrink (rings < 0) the mask by adjacency rings. */
  async dilateBodyMask(id: string, rings: number): Promise<BodyMask> {
    return json(
      await fetch(`/api/sessions/${id}/body_mask/dilate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rings }),
      }),
    )
  },

  bodyMaskUrl(id: string): string {
    return `/api/sessions/${id}/body_mask.json`
  },

  /** Persist the refit working state (placement/pins/frozen/preset/knobs). */
  async putRefitState(id: string, refit_state: RefitState | null): Promise<void> {
    await fetch(`/api/sessions/${id}/refit_state`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refit_state }),
    })
  },

  wrapUrl(id: string): string {
    return `/api/sessions/${id}/wrap.obj`
  },

  resultUrl(id: string, poseId: string): string {
    return `/api/sessions/${id}/results/${poseId}.obj`
  },

  zipUrl(id: string): string {
    return `/api/sessions/${id}/results.zip`
  },

  fbxUrl(id: string): string {
    return `/api/sessions/${id}/results.fbx`
  },

  async getQueueStatus(): Promise<QueueStatus> {
    return json(await fetch(`/api/queue`))
  },
}

export interface QueueStatus {
  waiting: number
  running: number
  concurrency: number
}
