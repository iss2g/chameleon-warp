import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import { MeshData } from '../api'

// 4x4 identity as a row-major 16-float array (the pre_transform the backend
// expects when the garment is left where it sits).
const IDENTITY16 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

/**
 * Orbit-camera state in mesh-local terms. We export/import this so two
 * viewports of different-sized meshes can stay rotationally synced (the
 * `distanceFactor` is normalized to each viewport's own bounding radius).
 */
export interface CameraState {
  azimuth: number       // spherical theta
  polar: number         // spherical phi
  distanceFactor: number  // distance / boundingSphereRadius
}

export interface MeshViewportHandle {
  frame: () => void
  getCameraState: () => CameraState | null
  applyCameraState: (state: CameraState, suppressEvents?: boolean) => void
}

interface Props {
  title: string
  mesh: MeshData | null
  markedVertexIndices?: number[]
  pendingVertex?: number | null
  emptyHint?: string
  /** One-line "what a click does right now", pinned to the bottom of the
   *  viewport. The wizard step supplies it, so the instruction sits where the
   *  action happens instead of only in the sidebar. Paint/fly HUDs win. */
  hudHint?: string | null
  onPickVertex?: (vertexIdx: number) => void
  /** Right-click on a marked vertex → remove that marker (index into
   *  markedVertexIndices, which matches the marker list order). */
  onRemoveMarker?: (markedListIndex: number) => void
  /** Right-click on the pending (yellow) pick → cancel it. */
  onCancelPending?: () => void
  onDropFile?: (file: File) => void
  meshColor?: number
  /** Per-vertex error heatmap: distance to the target surface as a fraction
   *  of its bbox diagonal (-1 = no data → base color). Null/undefined = off. */
  heatValuesFrac?: number[] | null
  /** Per-vertex layer id from auto layer mode (0 driven / 1 follower-wall /
   *  2 follower-component). When present it takes precedence over the grip
   *  heatmap: driven keeps the base color, follower = blue, component = purple.
   *  Null/undefined = off. */
  layerValues?: number[] | null
  /** Fraction mapped to full red. Default 0.02 (= 2% of the bbox diagonal,
   *  matching the default match/projection distance cut). */
  heatScaleFrac?: number
  /** Overlay the triangle topology (wireframe) on top of the shaded mesh. */
  wireframe?: boolean
  /** Fires whenever the OrbitControls state changes due to user input. */
  onUserCameraChange?: () => void
  // ---- region (partial-wrap) overlays ----
  boundaryWaypoints?: number[]    // clicked boundary points (orange, start magenta)
  boundaryClosed?: boolean        // draw the waypoint polyline closed vs open
  boundaryLoop?: number[]         // dense stitched loop, drawn as a closed line
  seedVertex?: number | null      // interior seed (green)
  regionEditableIndices?: number[]  // editable interior tint (faint green)
  // ---- refit place/resize gizmo ----
  /** Attach a TransformControls gizmo to THIS viewport's mesh so the user can
   *  move / rotate / scale it before a solve. 'off' = no gizmo. */
  transformMode?: 'off' | 'translate' | 'rotate' | 'scale'
  /** Fires (on drag end) with the mesh's current placement as a row-major 4x4
   *  (16 floats) — feed straight to refit's `pre_transform`. */
  onTransformChange?: (matrix16: number[]) => void
  /** Bump this to snap the gizmo/mesh back to identity (Reset placement). */
  transformResetToken?: number
  /** Placement to restore whenever the mesh is (re)built — row-major 4x4 (16
   *  floats). Survives viewport remounts (e.g. toggling the Preview tab) so the
   *  user's move/resize isn't lost. null/identity = leave at rest. */
  initialTransform?: number[] | null
  /** A second reference mesh drawn in the same world space (the body, so the
   *  garment can be placed / previewed against it). */
  ghostMesh?: MeshData | null
  /** How to render the ghost. 'ghost' (default) = semi-transparent, doesn't
   *  occlude — for placement in the editor. 'solid' = opaque, honest preview
   *  where the garment penetrates the body shows through. */
  ghostStyle?: 'ghost' | 'solid'
  /** Vertex indices of the ghost mesh that are "masked" (covered by the
   *  garment). When ghostMaskApply is true, ghost faces whose three vertices
   *  are all masked are dropped — hiding the covered skin. */
  ghostMaskVerts?: number[] | null
  ghostMaskApply?: boolean
  // ---- refit pins ("needles") ----
  /** When true, a left-click pokes a needle through the combined source+body
   *  overlay. Mutually exclusive with the transform gizmo. */
  pinMode?: boolean
  /** Fires with a placed pin. `marker` (inner source vtx ↔ nearest body vtx)
   *  is a hard correspondence — null if the needle didn't reach the body.
   *  `preserve` = the outer source verts pierced (release, keep their shape). */
  onAddPin?: (pin: { marker: { source: number; target: number } | null; preserve: number[] }) => void
  /** Needle length as a fraction of the mesh's world diameter (default 0.12). */
  pinDepthFrac?: number
  /** Highlight the preserve (outer-wall) verts of placed pins. */
  pinPreserve?: number[]
  // ---- frozen paint (refit) ----
  /** When true, left-drag brushes vertices as frozen and right-drag erases;
   *  OrbitControls are disabled for the duration. Exclusive with pins/gizmo. */
  paintMode?: boolean
  /** Brush radius as a fraction of the mesh's world diameter (default 0.04). */
  paintRadiusFrac?: number
  /** Fires per brush sample with the vertex indices under the brush.
   *  erase=true when the right button is painting. */
  onPaintStroke?: (verts: number[], erase: boolean) => void
  /** Highlight the frozen-painted verts (ice blue). */
  frozenVerts?: number[]
}

// Error-heatmap gradient: blue (on the target surface) → green → yellow →
// red (at/beyond the scale distance). Perceptually ordered, dark-bg friendly.
const HEAT_STOPS: Array<[number, THREE.Color]> = [
  [0.0, new THREE.Color(0x3566d6)],
  [0.5, new THREE.Color(0x3fbf6f)],
  [0.75, new THREE.Color(0xe8d44a)],
  [1.0, new THREE.Color(0xe04a3a)],
]

function heatColor(t: number, out: THREE.Color) {
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    if (t <= HEAT_STOPS[i][0]) {
      const [t0, c0] = HEAT_STOPS[i - 1]
      const [t1, c1] = HEAT_STOPS[i]
      out.copy(c0).lerp(c1, (t - t0) / (t1 - t0))
      return
    }
  }
  out.copy(HEAT_STOPS[HEAT_STOPS.length - 1][1])
}

// One stable color per marker index (cycles)
const MARKER_PALETTE = [
  '#81a73e', '#394b7c', '#aa8639', '#963968',
  '#307b62', '#4f2c73', '#aac139', '#b45f43',
  '#3b8f9c', '#a83d3d', '#5e8b3a', '#3a5ea3',
  '#b3873a', '#76408a', '#3a8a73', '#a85a8d',
]
export const markerColor = (idx: number): string => MARKER_PALETTE[idx % MARKER_PALETTE.length]

// Smallest sphere enclosing two spheres (used to frame the camera on the union
// of the main mesh and the body ghost when both are shown).
function mergeSpheres(a: THREE.Sphere, b: THREE.Sphere): THREE.Sphere {
  const center = a.center.clone()
  let radius = a.radius
  const d = center.distanceTo(b.center)
  if (d + b.radius <= radius) return new THREE.Sphere(center, radius)   // b ⊆ a
  if (d + radius <= b.radius) return new THREE.Sphere(b.center.clone(), b.radius) // a ⊆ b
  const newR = (radius + d + b.radius) / 2
  if (d > 1e-9) center.add(b.center.clone().sub(center).multiplyScalar((newR - radius) / d))
  radius = newR
  return new THREE.Sphere(center, radius)
}

// The sphere to frame the camera on: the main mesh, widened to include the
// ghost body when one is passed (caller passes null to keep it mesh-only).
function frameSphereFor(
  meshObj: THREE.Mesh | null,
  ghostObj: THREE.Mesh | null,
): THREE.Sphere | null {
  const ms = meshObj?.geometry.boundingSphere
  if (!ms) return null
  const gs = ghostObj?.geometry.boundingSphere
  return gs ? mergeSpheres(ms, gs) : ms.clone()
}

const MeshViewport = forwardRef<MeshViewportHandle, Props>(function MeshViewport(
  {
    title,
    mesh,
    markedVertexIndices = [],
    pendingVertex = null,
    emptyHint = '',
    hudHint = null,
    onPickVertex,
    onRemoveMarker,
    onCancelPending,
    onDropFile,
    meshColor,
    heatValuesFrac = null,
    layerValues = null,
    heatScaleFrac = 0.02,
    wireframe = false,
    onUserCameraChange,
    boundaryWaypoints = [],
    boundaryClosed = false,
    boundaryLoop,
    seedVertex = null,
    regionEditableIndices,
    transformMode = 'off',
    onTransformChange,
    transformResetToken = 0,
    initialTransform = null,
    ghostMesh = null,
    ghostStyle = 'ghost',
    ghostMaskVerts = null,
    ghostMaskApply = false,
    pinMode = false,
    onAddPin,
    pinDepthFrac = 0.12,
    pinPreserve = [],
    paintMode = false,
    paintRadiusFrac = 0.04,
    onPaintStroke,
    frozenVerts = [],
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    controls: OrbitControls
    transformControls: TransformControls | null
    // The gizmo transforms THIS group; the mesh, its wireframe and all marker /
    // region overlays are its children, so they move/scale together and the
    // placement survives mesh swaps. Restored from `initialTransform` on mount.
    placementGroup: THREE.Group
    ghostObject: THREE.Mesh | null
    meshObject: THREE.Mesh | null
    wireObject: THREE.LineSegments | null
    markerGroup: THREE.Group
    // Brush cursor ring drawn on the surface in paint mode.
    brushCursor: THREE.Mesh | null
    raycaster: THREE.Raycaster
    pointer: THREE.Vector2
    resizeObserver: ResizeObserver | null
    rafId: number
    disposed: boolean
  } | null>(null)

  // Fly-mode HUD state (Blender-style Shift+` navigation while painting).
  const [flyOn, setFlyOn] = useState(false)

  // Latest callback in a ref so the event listeners pick it up without re-binding.
  const onPickRef = useRef(onPickVertex)
  onPickRef.current = onPickVertex
  const onRemoveMarkerRef = useRef(onRemoveMarker)
  onRemoveMarkerRef.current = onRemoveMarker
  const onCancelPendingRef = useRef(onCancelPending)
  onCancelPendingRef.current = onCancelPending
  const onDropRef = useRef(onDropFile)
  onDropRef.current = onDropFile
  // Right-click removal reads these (current) values without re-binding the
  // one-time pointer listeners.
  const markedRef = useRef(markedVertexIndices)
  markedRef.current = markedVertexIndices
  const pendingRef = useRef(pendingVertex)
  pendingRef.current = pendingVertex
  const onUserCameraChangeRef = useRef(onUserCameraChange)
  onUserCameraChangeRef.current = onUserCameraChange
  const onTransformChangeRef = useRef(onTransformChange)
  onTransformChangeRef.current = onTransformChange
  const transformModeRef = useRef(transformMode)
  transformModeRef.current = transformMode
  const pinModeRef = useRef(pinMode)
  pinModeRef.current = pinMode
  const onAddPinRef = useRef(onAddPin)
  onAddPinRef.current = onAddPin
  const pinDepthFracRef = useRef(pinDepthFrac)
  pinDepthFracRef.current = pinDepthFrac
  const paintModeRef = useRef(paintMode)
  paintModeRef.current = paintMode
  const paintRadiusFracRef = useRef(paintRadiusFrac)
  paintRadiusFracRef.current = paintRadiusFrac
  const onPaintStrokeRef = useRef(onPaintStroke)
  onPaintStrokeRef.current = onPaintStroke
  // Only a SOLID ghost (the preview body) shares the mesh's coordinate frame and
  // should widen the framing. The editor's semi-transparent ghost sits under a
  // gizmo transform (different frame), so framing stays mesh-only there.
  const ghostStyleRef = useRef(ghostStyle)
  ghostStyleRef.current = ghostStyle
  // Read inside the mesh-build effect WITHOUT making it a dep (else every drag
  // would rebuild the mesh); the effect restores this placement on (re)build.
  const initialTransformRef = useRef(initialTransform)
  initialTransformRef.current = initialTransform

  // Suppress firing onUserCameraChange when we're applying an external sync update.
  const suppressChangeRef = useRef(false)

  // -------- one-time three.js setup --------
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#1d1f24')

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000)
    camera.position.set(0, 0, 3)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.addEventListener('change', () => {
      if (suppressChangeRef.current) return
      // Notify the parent so sync mode can mirror to the other viewport.
      onUserCameraChangeRef.current?.()
    })

    // Transform gizmo (refit place/resize). Created up front but hidden until a
    // caller sets transformMode !== 'off' and a mesh exists. While the user
    // drags a handle we disable OrbitControls so the camera doesn't spin, and
    // on drag end we emit the mesh's placement as a row-major 4x4.
    const transformControls = new TransformControls(camera, renderer.domElement)
    transformControls.visible = false
    transformControls.enabled = false
    scene.add(transformControls)

    // The placement group: the gizmo drives this, and mesh + overlays are its
    // children (added below / in the mesh effect). Restore any prior placement
    // (row-major, so set() loads it straight) so it survives viewport remounts.
    const placementGroup = new THREE.Group()
    const it0 = initialTransformRef.current
    if (it0 && it0.length === 16) {
      new THREE.Matrix4().set(
        it0[0], it0[1], it0[2], it0[3], it0[4], it0[5], it0[6], it0[7],
        it0[8], it0[9], it0[10], it0[11], it0[12], it0[13], it0[14], it0[15],
      ).decompose(placementGroup.position, placementGroup.quaternion, placementGroup.scale)
    }
    scene.add(placementGroup)

    const emitTransform = () => {
      const st = stateRef.current
      if (!st) return
      st.placementGroup.updateMatrix()
      // THREE stores elements column-major; the backend reshape(4,4) wants
      // row-major, so transpose before flattening.
      const m = st.placementGroup.matrix.clone().transpose().toArray()
      onTransformChangeRef.current?.(m)
    }
    transformControls.addEventListener('dragging-changed', (e) => {
      controls.enabled = !e.value
      if (!e.value) emitTransform()   // commit on drag end
    })

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.5))
    const key = new THREE.DirectionalLight(0xffffff, 0.8)
    key.position.set(2, 3, 4)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.3)
    fill.position.set(-3, -1, -2)
    scene.add(fill)

    // Marker / region overlays live UNDER the placement group so they follow
    // the mesh when it's moved/scaled (they're built from local vertex coords).
    const markerGroup = new THREE.Group()
    placementGroup.add(markerGroup)

    // Brush cursor: a flat ring positioned on the surface at the raycast hit,
    // oriented to the surface normal, scaled to the brush radius. Lives in
    // world space (not the placement group) since we set its world transform
    // directly from the hit. Hidden until paint mode moves it.
    const brushRing = new THREE.Mesh(
      new THREE.RingGeometry(0.86, 1.0, 48),
      new THREE.MeshBasicMaterial({
        color: 0x7fd8ff, transparent: true, opacity: 0.9,
        side: THREE.DoubleSide, depthTest: false,
      }),
    )
    brushRing.renderOrder = 20
    brushRing.visible = false
    scene.add(brushRing)

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()

    const fit = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (w === 0 || h === 0) return
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h, false)
    }
    fit()

    const ro = new ResizeObserver(fit)
    ro.observe(container)

    // ---- fly navigation (Blender-style Shift+`) ----
    // Keyboard flies the camera so the mouse stays free to paint; mouse-look
    // uses pointer lock. Toggled by Shift+Backquote while the pointer is over
    // this viewport, or Esc to exit. Movement/look are applied per frame.
    const fly = {
      on: false,
      keys: new Set<string>(),
      yaw: 0, pitch: 0,
      locked: false,
      lastT: 0,
    }
    const sphereRadius = () =>
      stateRef.current?.meshObject?.geometry?.boundingSphere?.radius ?? 1

    const syncYawPitchFromCamera = () => {
      const e = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
      fly.yaw = e.y
      fly.pitch = e.x
    }
    const setFlyModeInternal = (on: boolean) => {
      if (fly.on === on) return
      fly.on = on
      setFlyOn(on)
      if (on) {
        controls.enabled = false
        syncYawPitchFromCamera()
        renderer.domElement.requestPointerLock?.()
      } else {
        fly.keys.clear()
        if (document.pointerLockElement === renderer.domElement) document.exitPointerLock?.()
        // Resume orbit around a point in front of where we ended up.
        const dir = new THREE.Vector3()
        camera.getWorldDirection(dir)
        controls.target.copy(camera.position).addScaledVector(dir, sphereRadius() * 2)
        controls.enabled = true
        controls.update()
      }
    }
    const onLockChange = () => {
      fly.locked = document.pointerLockElement === renderer.domElement
      // If the user pressed Esc, the browser drops the lock — mirror to state.
      if (!fly.locked && fly.on) setFlyModeInternal(false)
    }
    document.addEventListener('pointerlockchange', onLockChange)
    const onFlyMouse = (e: MouseEvent) => {
      if (!fly.on || !fly.locked) return
      const s = 0.0022
      fly.yaw -= e.movementX * s
      fly.pitch -= e.movementY * s
      const lim = Math.PI / 2 - 0.01
      fly.pitch = Math.max(-lim, Math.min(lim, fly.pitch))
      const q = new THREE.Euler(fly.pitch, fly.yaw, 0, 'YXZ')
      camera.quaternion.setFromEuler(q)
    }
    document.addEventListener('mousemove', onFlyMouse)

    let rafId = 0
    const loop = () => {
      const st = stateRef.current
      if (fly.on) {
        const now = performance.now()
        const dt = fly.lastT ? Math.min(0.05, (now - fly.lastT) / 1000) : 0
        fly.lastT = now
        const base = sphereRadius() * 1.4 * (fly.keys.has('shiftleft') || fly.keys.has('shiftright') ? 3 : 1)
        const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd)
        const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize()
        const v = new THREE.Vector3()
        if (fly.keys.has('keyw')) v.addScaledVector(fwd, 1)
        if (fly.keys.has('keys')) v.addScaledVector(fwd, -1)
        if (fly.keys.has('keyd')) v.addScaledVector(right, 1)
        if (fly.keys.has('keya')) v.addScaledVector(right, -1)
        if (fly.keys.has('keye')) v.y += 1
        if (fly.keys.has('keyq')) v.y -= 1
        if (v.lengthSq() > 0) camera.position.addScaledVector(v.normalize(), base * dt)
      } else {
        fly.lastT = 0
        controls.update()
      }
      if (st) renderer.render(scene, camera)
      rafId = requestAnimationFrame(loop)
    }
    rafId = requestAnimationFrame(loop)

    stateRef.current = {
      renderer, scene, camera, controls,
      transformControls,
      placementGroup,
      ghostObject: null,
      meshObject: null,
      wireObject: null,
      markerGroup,
      brushCursor: brushRing,
      raycaster,
      pointer,
      resizeObserver: ro,
      rafId,
      disposed: false,
    }

    // ---- picking ----
    // Track movement between mousedown and mouseup so dragging the camera
    // doesn't register as a pick.
    let downX = 0
    let downY = 0
    let downTime = 0
    const onPointerDown = (e: PointerEvent) => {
      downX = e.clientX
      downY = e.clientY
      downTime = performance.now()
    }
    const onPointerUp = (e: PointerEvent) => {
      const dx = e.clientX - downX
      const dy = e.clientY - downY
      const dt = performance.now() - downTime
      // ~5px tolerance and < 350ms — treat as a click (anything more is a
      // camera drag: rotate on left, pan on right — don't pick/remove).
      if (dx * dx + dy * dy > 25 || dt > 350) return

      const st = stateRef.current
      if (!st || !st.meshObject) return
      // Frozen-paint mode owns the pointer entirely (strokes are handled by
      // their own listeners) — no picks, no marker removal.
      if (paintModeRef.current) return
      // Pin ("needle") mode: a left-click pokes through the layers instead of
      // placing a marker. Right-click does nothing here (remove via the panel).
      if (pinModeRef.current) {
        if (e.button === 0 && onAddPinRef.current) handleNeedlePick(e, st)
        return
      }
      // While the place/resize gizmo is active, the canvas is for dragging
      // handles — don't let a click on the mesh drop a stray marker.
      if (transformModeRef.current !== 'off') return

      // Right button → remove the nearest marker / cancel the pending pick.
      if (e.button === 2) {
        handleRightPick(e, st)
        return
      }
      // Only the primary (left) button places picks. Middle/other do nothing.
      if (e.button !== 0 || !onPickRef.current) return

      const rect = renderer.domElement.getBoundingClientRect()
      st.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      st.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      st.raycaster.setFromCamera(st.pointer, st.camera)
      const hits = st.raycaster.intersectObject(st.meshObject, false)
      if (hits.length === 0) return
      const hit = hits[0]
      if (!hit.face) return
      // Pick the triangle vertex nearest to the hit point
      const geom = (st.meshObject.geometry as THREE.BufferGeometry)
      const pos = geom.attributes.position as THREE.BufferAttribute
      const candidates = [hit.face.a, hit.face.b, hit.face.c]
      let bestIdx = candidates[0]
      let bestDist = Infinity
      const p = new THREE.Vector3()
      for (const idx of candidates) {
        p.fromBufferAttribute(pos, idx).applyMatrix4(st.meshObject.matrixWorld)
        const d = p.distanceToSquared(hit.point)
        if (d < bestDist) {
          bestDist = d
          bestIdx = idx
        }
      }
      onPickRef.current(bestIdx)
    }

    // Right-click removal: find the on-screen-nearest highlighted point
    // (a placed marker or the pending pick) within a small pixel radius, make
    // sure it isn't hidden behind the mesh, and fire the matching callback.
    const handleRightPick = (e: PointerEvent, st: NonNullable<typeof stateRef.current>) => {
      const removeCb = onRemoveMarkerRef.current
      const cancelCb = onCancelPendingRef.current
      if ((!removeCb || markedRef.current.length === 0) &&
          (!cancelCb || pendingRef.current === null)) return
      const meshObj = st.meshObject
      if (!meshObj) return

      const rect = st.renderer.domElement.getBoundingClientRect()
      const clickX = e.clientX - rect.left
      const clickY = e.clientY - rect.top
      const w = rect.width
      const h = rect.height
      const geom = meshObj.geometry as THREE.BufferGeometry
      const pos = geom.attributes.position as THREE.BufferAttribute
      const cam = st.camera
      const tmp = new THREE.Vector3()

      type RPick = { kind: 'marker' | 'pending'; listIndex: number; world: THREE.Vector3; d2: number }
      let best: RPick | null = null
      const consider = (kind: 'marker' | 'pending', listIndex: number, vIdx: number | null) => {
        if (vIdx === null || vIdx < 0 || vIdx >= pos.count) return
        tmp.fromBufferAttribute(pos, vIdx).applyMatrix4(meshObj.matrixWorld)
        const proj = tmp.clone().project(cam)
        if (proj.z < -1 || proj.z > 1) return // behind camera / clipped
        const sx = (proj.x * 0.5 + 0.5) * w
        const sy = (-proj.y * 0.5 + 0.5) * h
        const d2 = (sx - clickX) ** 2 + (sy - clickY) ** 2
        if (!best || d2 < best.d2) best = { kind, listIndex, world: tmp.clone(), d2 }
      }

      if (removeCb) markedRef.current.forEach((vIdx, j) => consider('marker', j, vIdx))
      if (cancelCb) consider('pending', -1, pendingRef.current)

      const THRESH = 14 // px — generous around the ~9–14px dots
      // `best` is mutated inside the consider() closure, which TS's flow
      // analysis can't follow (it still thinks best is null here) — the `as`
      // cast restores the real union type so the guard narrows correctly.
      const pick = best as RPick | null
      if (!pick || pick.d2 > THRESH * THRESH) return

      // Occlusion guard: only remove a point the user can actually see. Cast a
      // ray from the camera toward the point; if the mesh is hit clearly in
      // front of it, the dot is on the far side of the head — ignore.
      const camDist = cam.position.distanceTo(pick.world)
      const ndc = pick.world.clone().project(cam)
      st.raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), cam)
      const hits = st.raycaster.intersectObject(meshObj, false)
      const r = geom.boundingSphere?.radius ?? 1
      const eps = Math.max(1e-4, r * 0.02)
      if (hits.length > 0 && hits[0].distance < camDist - eps) return // occluded

      if (pick.kind === 'pending') cancelCb?.()
      else removeCb?.(pick.listIndex)
    }

    // Needle pick: cast the camera ray, collect EVERY source layer it pierces
    // within a fixed depth, plus the body-ghost hits. The source layer nearest
    // a body hit (or the deepest, if the needle doesn't reach the body) is the
    // inner wall → grip; the rest are outer → release. Reports source vertices.
    const handleNeedlePick = (e: PointerEvent, st: NonNullable<typeof stateRef.current>) => {
      const mesh = st.meshObject
      if (!mesh) return
      const rect = st.renderer.domElement.getBoundingClientRect()
      st.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      st.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      st.raycaster.setFromCamera(st.pointer, st.camera)
      const srcHits = st.raycaster.intersectObject(mesh, false)
      if (srcHits.length === 0) return
      const bodyHits = st.ghostObject
        ? st.raycaster.intersectObject(st.ghostObject, false) : []

      // Needle span: from the first SOURCE hit (the garment surface we poked)
      // forward `L` world units — NOT from the first body hit, since the body
      // isn't always in front of the source. Body hits are used unfiltered so
      // the marker can bind to the nearest one either side.
      const geom = mesh.geometry as THREE.BufferGeometry
      if (!geom.boundingSphere) geom.computeBoundingSphere()
      const s = st.placementGroup.scale
      const worldDiam = (geom.boundingSphere?.radius ?? 1) * 2 * Math.max(s.x, s.y, s.z)
      const L = Math.max(1e-6, pinDepthFracRef.current) * worldDiam
      const tStart = srcHits[0].distance
      const sHits = srcHits.filter((h) => h.distance <= tStart + L + 1e-9)
      if (sHits.length === 0) return
      const bHits = bodyHits

      // inner = source hit nearest (along the ray) to a body hit; if the needle
      // never reaches the body, the deepest source hit.
      let innerHit = sHits[sHits.length - 1]
      if (bHits.length) {
        let best = Infinity
        for (const h of sHits) {
          let d = Infinity
          for (const b of bHits) d = Math.min(d, Math.abs(h.distance - b.distance))
          if (d < best) { best = d; innerHit = h }
        }
      }

      const pos = geom.attributes.position as THREE.BufferAttribute
      const tmp = new THREE.Vector3()
      const nearestVtx = (hit: THREE.Intersection): number => {
        const f = hit.face!
        const cands = [f.a, f.b, f.c]
        let bi = cands[0]
        let bd = Infinity
        for (const idx of cands) {
          tmp.fromBufferAttribute(pos, idx).applyMatrix4(mesh.matrixWorld)
          const d = tmp.distanceToSquared(hit.point)
          if (d < bd) { bd = d; bi = idx }
        }
        return bi
      }

      const inner = nearestVtx(innerHit)
      // Outer walls → preserve (release). Everything the needle pierced except
      // the inner wall that grips.
      const preserve: number[] = []
      for (const h of sHits) {
        if (h === innerHit || !h.face) continue
        const v = nearestVtx(h)
        if (v !== inner && !preserve.includes(v)) preserve.push(v)
      }

      // Inner source ↔ body = a hard MARKER, but only if the needle actually
      // reached the body (else this poke just annotates a preserve region).
      let marker: { source: number; target: number } | null = null
      if (bHits.length && st.ghostObject) {
        const bh = bHits.reduce((a, b) =>
          Math.abs(b.distance - innerHit.distance) < Math.abs(a.distance - innerHit.distance) ? b : a)
        if (bh.face) {
          const gpos = (st.ghostObject.geometry as THREE.BufferGeometry)
            .attributes.position as THREE.BufferAttribute
          const gcands = [bh.face.a, bh.face.b, bh.face.c]
          let gbi = gcands[0]
          let gbd = Infinity
          for (const idx of gcands) {
            tmp.fromBufferAttribute(gpos, idx).applyMatrix4(st.ghostObject.matrixWorld)
            const d = tmp.distanceToSquared(bh.point)
            if (d < gbd) { gbd = d; gbi = idx }
          }
          marker = { source: inner, target: gbi }
        }
      }
      onAddPinRef.current?.({ marker, preserve })
    }

    // ---- frozen paint (refit) ----
    // Left-drag brushes vertices, right-drag erases. World positions are baked
    // once per stroke (mesh and camera are static while painting — orbit is
    // disabled by the paintMode effect), then each sample is a brute-force
    // radius scan around the raycast hit.
    let painting = false
    let paintErase = false
    let paintWorldPos: Float32Array | null = null
    // Move + size the brush cursor to a raycast hit; returns the hit (or null).
    const showBrushAt = (e: PointerEvent): THREE.Intersection | null => {
      const st = stateRef.current
      if (!st || !st.meshObject || !st.brushCursor) return null
      const rect = renderer.domElement.getBoundingClientRect()
      st.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      st.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      st.raycaster.setFromCamera(st.pointer, st.camera)
      const hits = st.raycaster.intersectObject(st.meshObject, false)
      if (hits.length === 0) { st.brushCursor.visible = false; return null }
      const hit = hits[0]
      const geom = st.meshObject.geometry as THREE.BufferGeometry
      if (!geom.boundingSphere) geom.computeBoundingSphere()
      const s = st.placementGroup.scale
      const worldDiam = (geom.boundingSphere?.radius ?? 1) * 2 * Math.max(s.x, s.y, s.z)
      const radius = Math.max(1e-6, paintRadiusFracRef.current) * worldDiam
      // Orient the ring's +Z to the surface normal (world space), place a hair
      // above the surface so it doesn't z-fight, scale to the brush radius.
      const wn = hit.face
        ? hit.face.normal.clone()
            .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(st.meshObject.matrixWorld))
            .normalize()
        : new THREE.Vector3(0, 0, 1)
      st.brushCursor.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), wn)
      st.brushCursor.position.copy(hit.point).addScaledVector(wn, radius * 0.02)
      st.brushCursor.scale.setScalar(radius)
      ;(st.brushCursor.material as THREE.MeshBasicMaterial).color.set(
        paintErase ? 0xff7a7a : 0x7fd8ff)
      st.brushCursor.visible = true
      return hit
    }
    const samplePaint = (e: PointerEvent) => {
      const st = stateRef.current
      if (!st || !st.meshObject) return
      const hit = showBrushAt(e)
      if (!hit) return
      const geom = st.meshObject.geometry as THREE.BufferGeometry
      const pos = geom.attributes.position as THREE.BufferAttribute
      if (!paintWorldPos) {
        paintWorldPos = new Float32Array(pos.count * 3)
        const v = new THREE.Vector3()
        for (let i = 0; i < pos.count; i++) {
          v.fromBufferAttribute(pos, i).applyMatrix4(st.meshObject.matrixWorld)
          paintWorldPos[i * 3] = v.x
          paintWorldPos[i * 3 + 1] = v.y
          paintWorldPos[i * 3 + 2] = v.z
        }
      }
      const s = st.placementGroup.scale
      const worldDiam = (geom.boundingSphere?.radius ?? 1) * 2 * Math.max(s.x, s.y, s.z)
      const r2 = (Math.max(1e-6, paintRadiusFracRef.current) * worldDiam) ** 2
      const found: number[] = []
      for (let i = 0; i < pos.count; i++) {
        const dx = paintWorldPos[i * 3] - hit.point.x
        const dy = paintWorldPos[i * 3 + 1] - hit.point.y
        const dz = paintWorldPos[i * 3 + 2] - hit.point.z
        if (dx * dx + dy * dy + dz * dz <= r2) found.push(i)
      }
      if (found.length) onPaintStrokeRef.current?.(found, paintErase)
    }
    const onPaintDown = (e: PointerEvent) => {
      // Fly mode captures the mouse for look — no painting while flying.
      if (!paintModeRef.current || !onPaintStrokeRef.current || fly.on) return
      if (e.button !== 0 && e.button !== 2) return
      painting = true
      paintErase = e.button === 2
      paintWorldPos = null
      renderer.domElement.setPointerCapture(e.pointerId)
      samplePaint(e)
    }
    const onPaintMove = (e: PointerEvent) => {
      if (!paintModeRef.current || fly.on) return
      if (painting) samplePaint(e)
      else showBrushAt(e)          // hover: just position the cursor ring
    }
    const onPaintEnd = () => {
      painting = false
      paintWorldPos = null
    }
    const onPaintLeave = () => {
      painting = false
      paintWorldPos = null
      if (stateRef.current?.brushCursor) stateRef.current.brushCursor.visible = false
    }

    // ---- fly navigation key handling ----
    // Shift+` toggles fly mode while the pointer is over THIS viewport (so the
    // two side-by-side viewports don't both toggle). WASD/QE fly; Shift boosts.
    let hovered = false
    const onEnter = () => { hovered = true }
    const onLeave = () => { hovered = false }
    container.addEventListener('pointerenter', onEnter)
    container.addEventListener('pointerleave', onLeave)
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Backquote' && e.shiftKey && hovered) {
        e.preventDefault()
        setFlyModeInternal(!fly.on)
        return
      }
      if (!fly.on) return
      if (e.code === 'Escape') { setFlyModeInternal(false); return }
      fly.keys.add(e.code.toLowerCase())
    }
    const onKeyUp = (e: KeyboardEvent) => { fly.keys.delete(e.code.toLowerCase()) }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)

    // Suppress the browser context menu inside the canvas so right-click is
    // free for marker removal (OrbitControls also does this, but be explicit).
    const onContextMenu = (e: MouseEvent) => e.preventDefault()
    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)
    renderer.domElement.addEventListener('pointerdown', onPaintDown)
    renderer.domElement.addEventListener('pointermove', onPaintMove)
    renderer.domElement.addEventListener('pointerup', onPaintEnd)
    renderer.domElement.addEventListener('pointerleave', onPaintLeave)
    renderer.domElement.addEventListener('contextmenu', onContextMenu)

    // ---- drag-drop (only wire if a handler is provided) ----
    const onDragOver = (e: DragEvent) => {
      if (!onDropRef.current) return
      e.preventDefault()
      container.classList.add('drop-active')
    }
    const onDragLeave = () => container.classList.remove('drop-active')
    const onDrop = (e: DragEvent) => {
      if (!onDropRef.current) return
      e.preventDefault()
      container.classList.remove('drop-active')
      const file = e.dataTransfer?.files?.[0]
      if (file) onDropRef.current(file)
    }
    container.addEventListener('dragover', onDragOver)
    container.addEventListener('dragleave', onDragLeave)
    container.addEventListener('drop', onDrop)

    return () => {
      const st = stateRef.current
      if (st) {
        st.disposed = true
        cancelAnimationFrame(st.rafId)
        st.resizeObserver?.disconnect()
        st.controls.dispose()
        if (st.transformControls) {
          st.transformControls.detach()
          st.scene.remove(st.transformControls)
          st.transformControls.dispose()
        }
        if (st.ghostObject) {
          st.scene.remove(st.ghostObject)
          st.ghostObject.geometry.dispose()
          ;(st.ghostObject.material as THREE.Material).dispose()
        }
        if (st.brushCursor) {
          st.scene.remove(st.brushCursor)
          st.brushCursor.geometry.dispose()
          ;(st.brushCursor.material as THREE.Material).dispose()
        }
        if (document.pointerLockElement === st.renderer.domElement) document.exitPointerLock?.()
        st.renderer.dispose()
        if (st.renderer.domElement.parentNode) {
          st.renderer.domElement.parentNode.removeChild(st.renderer.domElement)
        }
      }
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      renderer.domElement.removeEventListener('pointerdown', onPaintDown)
      renderer.domElement.removeEventListener('pointermove', onPaintMove)
      renderer.domElement.removeEventListener('pointerup', onPaintEnd)
      renderer.domElement.removeEventListener('pointerleave', onPaintLeave)
      renderer.domElement.removeEventListener('contextmenu', onContextMenu)
      container.removeEventListener('pointerenter', onEnter)
      container.removeEventListener('pointerleave', onLeave)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      document.removeEventListener('pointerlockchange', onLockChange)
      document.removeEventListener('mousemove', onFlyMouse)
      container.removeEventListener('dragover', onDragOver)
      container.removeEventListener('dragleave', onDragLeave)
      container.removeEventListener('drop', onDrop)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Paint mode no longer hard-locks the camera (fly mode + middle-mouse orbit
  // cover navigation). We only remap the mouse buttons so left/right drag paint
  // instead of orbit: LEFT→paint (disabled in OrbitControls), MIDDLE→orbit,
  // RIGHT→erase (disabled in OrbitControls). Restored to defaults when off.
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    const mb = st.controls.mouseButtons
    if (paintMode) {
      mb.LEFT = null as unknown as THREE.MOUSE
      mb.MIDDLE = THREE.MOUSE.ROTATE
      mb.RIGHT = null as unknown as THREE.MOUSE
    } else {
      mb.LEFT = THREE.MOUSE.ROTATE
      mb.MIDDLE = THREE.MOUSE.DOLLY
      mb.RIGHT = THREE.MOUSE.PAN
    }
    st.controls.enabled = true
  }, [paintMode])

  // Tracks whether we've already framed the camera on a mesh in this viewport.
  // We only auto-frame the FIRST mesh — subsequent mesh swaps (animation
  // playback, swapping pose preview) keep the existing camera so the view
  // doesn't jump on every frame.
  const cameraFramedRef = useRef(false)
  // Whether we've auto-framed the union since the current ghost appeared. Reset
  // when the ghost goes away; kept across mask-toggle rebuilds so toggling the
  // body mask doesn't yank the camera.
  const ghostFramedRef = useRef(false)

  // -------- update mesh when prop changes --------
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    // Remove old mesh (and its wireframe overlay, rebuilt by its own effect)
    if (st.meshObject) {
      st.placementGroup.remove(st.meshObject)
      st.meshObject.geometry.dispose()
      ;(st.meshObject.material as THREE.Material).dispose()
      st.meshObject = null
    }
    if (st.wireObject) {
      st.scene.remove(st.wireObject)
      st.wireObject.geometry.dispose()
      ;(st.wireObject.material as THREE.Material).dispose()
      st.wireObject = null
    }
    if (!mesh) {
      cameraFramedRef.current = false
      return
    }

    const geom = new THREE.BufferGeometry()
    const positions = new Float32Array(mesh.vertices)
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh.faces), 1))
    geom.computeVertexNormals()
    geom.computeBoundingBox()
    geom.computeBoundingSphere()

    const mat = new THREE.MeshPhongMaterial({
      color: meshColor ?? 0xb8c0cc,
      side: THREE.DoubleSide,
      flatShading: false,
      shininess: 30,
      // Nudge the mesh slightly back in the depth buffer so markers placed
      // exactly ON vertices win the depth test on the visible side without
      // z-fight, while still being correctly occluded by triangles in front
      // of them when the user rotates the head.
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
    const obj = new THREE.Mesh(geom, mat)
    // Child of the placement group, so it inherits the gizmo's move/scale and
    // the placement persists across mesh swaps (the group is never rebuilt).
    st.placementGroup.add(obj)
    st.meshObject = obj

    if (!cameraFramedRef.current) {
      // First mesh: fit the camera around it (union with a solid body ghost if
      // one is already present, e.g. on a mesh swap).
      const sphere = frameSphereFor(
        obj, ghostStyle === 'solid' ? st.ghostObject : null) ?? geom.boundingSphere!
      const r = sphere.radius || 1
      st.controls.target.copy(sphere.center)
      st.camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + r * 3)
      st.camera.near = r * 0.01
      st.camera.far = r * 100
      st.camera.updateProjectionMatrix()
      st.controls.update()
      cameraFramedRef.current = true
    }
    // For subsequent meshes: leave camera/controls alone. The animation
    // assumption is that all frames share roughly the same world bounds.
  }, [mesh])

  // -------- error heatmap: per-vertex colors over the base material --------
  // Runs after the mesh effect (declared later, same [mesh] dep), so the
  // geometry is fresh. -1 values (frozen region) keep the base color; a
  // length mismatch means the values describe a different mesh → heatmap off.
  useEffect(() => {
    const st = stateRef.current
    const obj = st?.meshObject
    if (!st || !obj) return
    const geom = obj.geometry as THREE.BufferGeometry
    const mat = obj.material as THREE.MeshPhongMaterial
    const nVerts = (geom.attributes.position as THREE.BufferAttribute).count

    if (layerValues && layerValues.length === nVerts) {
      // Auto layer overlay (categorical) — wins over the grip heatmap.
      const base = new THREE.Color(meshColor ?? 0xb8c0cc)
      const follower = new THREE.Color(0x3566d6)   // blue = follower wall
      const component = new THREE.Color(0x9b59b6)  // purple = follower component
      const colors = new Float32Array(nVerts * 3)
      const c = new THREE.Color()
      for (let i = 0; i < nVerts; i++) {
        const l = layerValues[i]
        c.copy(l === 2 ? component : l === 1 ? follower : base)
        colors[i * 3] = c.r
        colors[i * 3 + 1] = c.g
        colors[i * 3 + 2] = c.b
      }
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      mat.vertexColors = true
      mat.color.set(0xffffff)
      mat.needsUpdate = true
    } else if (heatValuesFrac && heatValuesFrac.length === nVerts && heatScaleFrac > 0) {
      const base = new THREE.Color(meshColor ?? 0xb8c0cc)
      const colors = new Float32Array(nVerts * 3)
      const c = new THREE.Color()
      for (let i = 0; i < nVerts; i++) {
        const f = heatValuesFrac[i]
        if (f < 0) c.copy(base)
        else heatColor(Math.min(1, f / heatScaleFrac), c)
        colors[i * 3] = c.r
        colors[i * 3 + 1] = c.g
        colors[i * 3 + 2] = c.b
      }
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      mat.vertexColors = true
      mat.color.set(0xffffff)   // vertex colors carry the tint
      mat.needsUpdate = true
    } else {
      if (geom.hasAttribute('color')) geom.deleteAttribute('color')
      if (mat.vertexColors) {
        mat.vertexColors = false
        mat.color.set(meshColor ?? 0xb8c0cc)
        mat.needsUpdate = true
      }
    }
  }, [mesh, heatValuesFrac, heatScaleFrac, meshColor, layerValues])

  // -------- wireframe (topology) overlay --------
  // A LineSegments over the rendered triangles. The base material already
  // has polygonOffset pushing the surface back, so the lines sit cleanly on
  // top without z-fighting. Depth-tested, so the back side is occluded when
  // the mesh rotates. Runs after the mesh effect (same [mesh] dep, declared
  // later) so meshObject is current.
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    if (st.wireObject) {
      st.placementGroup.remove(st.wireObject)
      st.wireObject.geometry.dispose()
      ;(st.wireObject.material as THREE.Material).dispose()
      st.wireObject = null
    }
    if (!wireframe || !st.meshObject) return
    const wg = new THREE.WireframeGeometry(st.meshObject.geometry)
    const wm = new THREE.LineBasicMaterial({
      color: 0x11141a,
      transparent: true,
      opacity: 0.6,
      depthTest: true,
    })
    const wire = new THREE.LineSegments(wg, wm)
    wire.renderOrder = 1   // over the surface, under the marker dots
    st.placementGroup.add(wire)   // follow the placement transform too
    st.wireObject = wire
  }, [mesh, wireframe])

  // -------- refit place/resize gizmo: attach / detach + set mode --------
  // Runs after the mesh effect (same [mesh] dep, declared later) so meshObject
  // is current. A fresh mesh is created at identity, so swapping meshes also
  // resets any prior placement — intentional (you re-place per solve).
  useEffect(() => {
    const st = stateRef.current
    const tc = st?.transformControls
    if (!st || !tc) return
    if (transformMode !== 'off' && st.meshObject) {
      tc.attach(st.placementGroup)
      tc.setMode(transformMode)
      tc.visible = true
      tc.enabled = true
    } else {
      tc.detach()
      tc.visible = false
      tc.enabled = false
    }
  }, [mesh, transformMode])

  // -------- reset the placement to identity on demand (Reset button) --------
  // Skips the initial mount (token 0) so we don't fire an identity change before
  // the user has touched anything.
  const didMountResetRef = useRef(false)
  useEffect(() => {
    if (!didMountResetRef.current) {
      didMountResetRef.current = true
      return
    }
    const st = stateRef.current
    if (!st) return
    st.placementGroup.position.set(0, 0, 0)
    st.placementGroup.quaternion.identity()
    st.placementGroup.scale.set(1, 1, 1)
    st.placementGroup.updateMatrix()
    onTransformChangeRef.current?.(IDENTITY16.slice())
  }, [transformResetToken])

  // -------- ghost reference overlay (the body) --------
  // Two modes:
  //  'ghost' (editor placement): semi-transparent, non-depth-writing so it
  //    never occludes the garment being placed.
  //  'solid' (result preview): opaque, honest — where the garment penetrates
  //    the body it shows through. Optionally hides the faces covered by the
  //    garment (body mask), so only the exposed skin remains.
  // Always raycastable so a pin needle can find the body hit; the normal
  // left-pick only intersects the source mesh, so it's never stolen there.
  useEffect(() => {
    const st = stateRef.current
    if (!st) return
    if (st.ghostObject) {
      st.scene.remove(st.ghostObject)
      st.ghostObject.geometry.dispose()
      ;(st.ghostObject.material as THREE.Material).dispose()
      st.ghostObject = null
    }
    if (!ghostMesh) {
      ghostFramedRef.current = false
      return
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(ghostMesh.vertices), 3))

    // Face index: drop faces whose three verts are ALL masked when the mask is
    // applied (mirrors collision.hidden_faces). We rebuild locally from the
    // unfiltered faces each toggle — no re-fetch of the mesh.
    let faces = ghostMesh.faces
    if (ghostMaskApply && ghostMaskVerts && ghostMaskVerts.length) {
      const masked = new Uint8Array(ghostMesh.vertex_count)
      for (const v of ghostMaskVerts) if (v >= 0 && v < masked.length) masked[v] = 1
      const kept: number[] = []
      for (let i = 0; i < faces.length; i += 3) {
        const a = faces[i], b = faces[i + 1], c = faces[i + 2]
        if (masked[a] && masked[b] && masked[c]) continue
        kept.push(a, b, c)
      }
      faces = kept
    }
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1))
    geom.computeVertexNormals()
    geom.computeBoundingSphere()

    const solid = ghostStyle === 'solid'
    const mat = new THREE.MeshPhongMaterial({
      // Distinct neutral "before"-mesh color (matches the source-before panels).
      color: solid ? 0x8fa6c4 : 0x6a7686,
      side: THREE.DoubleSide,
      transparent: !solid,
      opacity: solid ? 1.0 : 0.28,
      depthWrite: solid,   // solid occludes honestly; ghost doesn't hide the garment
    })
    const obj = new THREE.Mesh(geom, mat)
    obj.renderOrder = solid ? 0 : -1     // ghost draws under the garment
    st.scene.add(obj)
    st.ghostObject = obj

    // When a solid body first appears, frame the union so it isn't half
    // off-screen. Only once per ghost — mask-toggle rebuilds keep the camera.
    if (solid && !ghostFramedRef.current && st.meshObject) {
      ghostFramedRef.current = true
      const sphere = frameSphereFor(st.meshObject, obj)
      if (sphere) {
        const r = sphere.radius || 1
        st.controls.target.copy(sphere.center)
        st.camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + r * 3)
        st.camera.near = r * 0.01
        st.camera.far = r * 100
        st.camera.updateProjectionMatrix()
        st.controls.update()
      }
    }
  }, [ghostMesh, ghostStyle, ghostMaskVerts, ghostMaskApply])

  // -------- imperative API: frame, getCameraState, applyCameraState --------
  useImperativeHandle(ref, () => ({
    frame() {
      const st = stateRef.current
      if (!st || !st.meshObject) return
      const sphere = frameSphereFor(
        st.meshObject, ghostStyleRef.current === 'solid' ? st.ghostObject : null)
      if (!sphere) return
      const r = sphere.radius || 1
      st.controls.target.copy(sphere.center)
      st.camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + r * 3)
      st.camera.near = r * 0.01
      st.camera.far = r * 100
      st.camera.updateProjectionMatrix()
      st.controls.update()
    },
    getCameraState() {
      const st = stateRef.current
      if (!st || !st.meshObject) return null
      const sphere = st.meshObject.geometry.boundingSphere
      const r = sphere?.radius || 1
      const offset = st.camera.position.clone().sub(st.controls.target)
      const spherical = new THREE.Spherical().setFromVector3(offset)
      return {
        azimuth: spherical.theta,
        polar: spherical.phi,
        distanceFactor: spherical.radius / r,
      }
    },
    applyCameraState(state, suppressEvents = true) {
      const st = stateRef.current
      if (!st || !st.meshObject) return
      const sphere = st.meshObject.geometry.boundingSphere
      const r = sphere?.radius || 1
      const sph = new THREE.Spherical(
        Math.max(0.0001, r * state.distanceFactor),
        Math.max(0.0001, Math.min(Math.PI - 0.0001, state.polar)),
        state.azimuth,
      )
      const offset = new THREE.Vector3().setFromSpherical(sph)
      // Keep the existing target; only move the camera around it.
      const prev = suppressChangeRef.current
      if (suppressEvents) suppressChangeRef.current = true
      st.camera.position.copy(st.controls.target).add(offset)
      st.controls.update()
      if (suppressEvents) suppressChangeRef.current = prev
    },
  }), [])

  // -------- update markers (and pending pick) --------
  //
  // Markers and the pending pick use THREE.Points with sizeAttenuation=false,
  // so they render at a constant pixel size no matter how far you zoom in.
  // This makes precise vertex picking on dense meshes painless: zoom in,
  // the dots stay small, you can see which vertex you're actually on.
  useEffect(() => {
    const st = stateRef.current
    if (!st || !mesh) return
    // Clear group
    while (st.markerGroup.children.length) {
      const c = st.markerGroup.children[0]
      st.markerGroup.remove(c)
      const anyc = c as THREE.Points
      if (anyc.geometry) anyc.geometry.dispose()
      const m = anyc.material
      if (m && !Array.isArray(m)) (m as THREE.Material).dispose()
    }

    const posAttr = st.meshObject?.geometry.attributes.position as THREE.BufferAttribute | undefined
    if (!posAttr) return

    // ---- Pinned markers as a Points cloud with vertex colors ----
    if (markedVertexIndices.length > 0) {
      const positions = new Float32Array(markedVertexIndices.length * 3)
      const colors = new Float32Array(markedVertexIndices.length * 3)
      const tmpColor = new THREE.Color()
      const tmpVec = new THREE.Vector3()

      let written = 0
      markedVertexIndices.forEach((vIdx, i) => {
        if (vIdx < 0 || vIdx >= posAttr.count) return
        tmpVec.fromBufferAttribute(posAttr, vIdx)
        positions[written * 3 + 0] = tmpVec.x
        positions[written * 3 + 1] = tmpVec.y
        positions[written * 3 + 2] = tmpVec.z
        tmpColor.set(markerColor(i))
        colors[written * 3 + 0] = tmpColor.r
        colors[written * 3 + 1] = tmpColor.g
        colors[written * 3 + 2] = tmpColor.b
        written++
      })

      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(positions.slice(0, written * 3), 3))
      g.setAttribute('color', new THREE.BufferAttribute(colors.slice(0, written * 3), 3))

      const mat = new THREE.PointsMaterial({
        size: 9,                 // pixels
        sizeAttenuation: false,  // constant on-screen size
        vertexColors: true,
        depthTest: true,         // occluded when the vertex faces away from camera
        transparent: true,
        opacity: 0.95,
      })
      const pts = new THREE.Points(g, mat)
      pts.renderOrder = 10
      st.markerGroup.add(pts)
    }

    // ---- Pending pick: a slightly larger, contrasting yellow dot ----
    if (pendingVertex !== null && pendingVertex >= 0 && pendingVertex < posAttr.count) {
      const g = new THREE.BufferGeometry()
      const p = new THREE.Vector3().fromBufferAttribute(posAttr, pendingVertex)
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([p.x, p.y, p.z]), 3))
      const mat = new THREE.PointsMaterial({
        size: 14,
        sizeAttenuation: false,
        color: 0xffe34d,
        depthTest: true,        // same occlusion rule as pinned markers
        transparent: true,
        opacity: 1.0,
      })
      const pts = new THREE.Points(g, mat)
      pts.renderOrder = 11
      st.markerGroup.add(pts)
    }

    // ---- Region overlays (partial-wrap boundary drawing) ----
    const gatherPositions = (indices: number[]): Float32Array => {
      const out = new Float32Array(indices.length * 3)
      const tmp = new THREE.Vector3()
      let w = 0
      for (const idx of indices) {
        if (idx < 0 || idx >= posAttr.count) continue
        tmp.fromBufferAttribute(posAttr, idx)
        out[w * 3] = tmp.x
        out[w * 3 + 1] = tmp.y
        out[w * 3 + 2] = tmp.z
        w++
      }
      return out.slice(0, w * 3)
    }
    const addPoints = (
      indices: number[], color: number, size: number, opacity: number, order: number,
    ) => {
      if (indices.length === 0) return
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(gatherPositions(indices), 3))
      const mat = new THREE.PointsMaterial({
        size, sizeAttenuation: false, color, depthTest: true,
        transparent: true, opacity,
      })
      const pts = new THREE.Points(g, mat)
      pts.renderOrder = order
      st.markerGroup.add(pts)
    }

    // Editable interior tint (faint green) — drawn first / under the rest.
    if (regionEditableIndices && regionEditableIndices.length > 0) {
      addPoints(regionEditableIndices, 0x4caf6a, 5, 0.5, 8)
    }
    // Boundary line. After a preview we have the dense stitched loop (always
    // closed). While drawing we show the raw waypoint polyline — OPEN until
    // the user closes it (clicking the start point), then closed.
    const hasDense = !!(boundaryLoop && boundaryLoop.length >= 2)
    const lineIdx = hasDense ? boundaryLoop! : boundaryWaypoints
    const drawClosed = hasDense || boundaryClosed
    if (lineIdx.length >= 2) {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(gatherPositions(lineIdx), 3))
      const mat = new THREE.LineBasicMaterial({
        color: 0x35d0e0, transparent: true, opacity: 0.95, depthTest: true,
      })
      const line = drawClosed
        ? new THREE.LineLoop(g, mat)   // closed: connect last back to first
        : new THREE.Line(g, mat)       // open polyline while still drawing
      line.renderOrder = 12
      st.markerGroup.add(line)
    }
    // Clicked waypoints. Highlight the START point (magenta) so the user knows
    // where to click to close the loop; the rest are orange. Once closed (or
    // previewed) there's nothing special to aim at, so colour them uniformly.
    if (!drawClosed && boundaryWaypoints.length > 0) {
      addPoints([boundaryWaypoints[0]], 0xff3ce0, 13, 1.0, 13)   // start = magenta
      addPoints(boundaryWaypoints.slice(1), 0xff9a3c, 10, 1.0, 13)
    } else {
      addPoints(boundaryWaypoints, 0xff9a3c, 10, 1.0, 13)
    }
    // Seed (bright green, large).
    if (seedVertex !== null && seedVertex >= 0) {
      addPoints([seedVertex], 0x46e06a, 15, 1.0, 14)
    }
    // Refit pin "preserve" (outer-wall) verts = orange. The inner grip shows as
    // a normal marker dot (it's a real marker in the session).
    if (pinPreserve.length > 0) addPoints(pinPreserve, 0xff8a3c, 11, 0.95, 15)
    // Frozen paint = ice blue. Drawn small — painted regions are dense.
    if (frozenVerts.length > 0) addPoints(frozenVerts, 0x7fd8ff, 7, 0.85, 14)
  }, [mesh, markedVertexIndices, pendingVertex,
      boundaryWaypoints, boundaryClosed, boundaryLoop, seedVertex, regionEditableIndices,
      pinPreserve, frozenVerts])

  const handleFrame = () => {
    const st = stateRef.current
    if (!st || !st.meshObject) return
    const sphere = frameSphereFor(
      st.meshObject, ghostStyleRef.current === 'solid' ? st.ghostObject : null)
    if (!sphere) return
    const r = sphere.radius || 1
    st.controls.target.copy(sphere.center)
    st.camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + r * 3)
    st.camera.near = r * 0.01
    st.camera.far = r * 100
    st.camera.updateProjectionMatrix()
    st.controls.update()
  }

  return (
    <div className="viewport-wrap">
      <div className="viewport-title">{title}</div>
      <div ref={containerRef} className="viewport">
        {!mesh && <div className="viewport-empty">{emptyHint}</div>}
        {mesh && (
          <div className="viewport-toolbar">
            <button
              className="viewport-btn"
              onClick={handleFrame}
              title="Frame mesh (reset camera)"
            >
              ⊡ Frame
            </button>
          </div>
        )}
        {hudHint && mesh && !paintMode && !flyOn && (
          <div className="viewport-hud step-hint">{hudHint}</div>
        )}
        {paintMode && !flyOn && (
          <div className="viewport-hud">
            paint: <b>L</b> draw · <b>R</b> erase · <b>MMB</b> orbit ·{' '}
            <b>Shift+`</b> fly
          </div>
        )}
        {flyOn && (
          <div className="viewport-hud fly">
            ✈ Fly — <b>WASD</b> move · <b>Q/E</b> down/up · <b>Shift</b> boost ·{' '}
            mouse look · <b>Esc</b> / <b>Shift+`</b> exit
          </div>
        )}
      </div>
    </div>
  )
})

export default MeshViewport
