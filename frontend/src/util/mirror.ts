/**
 * Find the vertex on `positions` that is the spatial mirror of vertex `idx`
 * across the chosen axis.
 *
 *   axis = 0 (X) → mirror across YZ plane → flip x        (standard face L↔R symmetry)
 *   axis = 1 (Y) → mirror across XZ plane → flip y
 *   axis = 2 (Z) → mirror across XY plane → flip z
 *
 * The mirror plane passes through `anchor` (default: world origin). For meshes
 * that are NOT centered at the origin you MUST pass the mesh centroid as the
 * anchor — otherwise the mirrored coordinates fly off into empty space and no
 * candidate vertex will be within the tolerance.
 *
 * Returns the vertex index whose position is closest to the mirrored
 * coordinates, or null if no candidate is within `tol`. If the closest match
 * is the same vertex (i.e. the vertex sits on the mirror plane), returns the
 * idx as-is — caller should treat that as a "self-mirror" and avoid creating
 * a duplicate marker.
 *
 * Linear scan: O(N). Called only when the user clicks a vertex, which is
 * infrequent enough that scanning ~50k verts (< 5 ms) is fine. A KD-tree
 * would be premature optimization.
 */
export function findMirrorVertex(
  positions: Float32Array | number[],
  idx: number,
  axis: 0 | 1 | 2,
  tol: number,
  anchor: [number, number, number] = [0, 0, 0],
): number | null {
  const N = positions.length / 3
  if (idx < 0 || idx >= N) return null
  const tx = positions[idx * 3 + 0]
  const ty = positions[idx * 3 + 1]
  const tz = positions[idx * 3 + 2]

  // Mirror across the plane that passes through `anchor` and is perpendicular
  // to the chosen axis: target = 2*anchor - source on that axis only.
  const mx = axis === 0 ? 2 * anchor[0] - tx : tx
  const my = axis === 1 ? 2 * anchor[1] - ty : ty
  const mz = axis === 2 ? 2 * anchor[2] - tz : tz

  let bestIdx = -1
  let bestDist = tol * tol
  for (let i = 0; i < N; i++) {
    const dx = positions[i * 3 + 0] - mx
    const dy = positions[i * 3 + 1] - my
    const dz = positions[i * 3 + 2] - mz
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 < bestDist) {
      bestDist = d2
      bestIdx = i
    }
  }
  return bestIdx >= 0 ? bestIdx : null
}

/**
 * Pick a reasonable mirror tolerance from the mesh bounding sphere radius.
 * Vertices within 1.5% of the radius are considered "the same mirror" — loose
 * enough to absorb floating-point noise on decimated/retopo'd meshes (which is
 * the common case for retargeting), tight enough to reject obvious mis-matches.
 */
export function defaultMirrorTolerance(boundingRadius: number): number {
  return Math.max(1e-5, boundingRadius * 0.015)
}

/** Mesh centroid: arithmetic mean of vertex positions. */
export function meshCentroid(positions: Float32Array | number[]): [number, number, number] {
  const N = positions.length / 3
  if (N === 0) return [0, 0, 0]
  let cx = 0, cy = 0, cz = 0
  for (let i = 0; i < N; i++) {
    cx += positions[i * 3 + 0]
    cy += positions[i * 3 + 1]
    cz += positions[i * 3 + 2]
  }
  return [cx / N, cy / N, cz / N]
}

/**
 * Bounding sphere radius around the centroid, in the same flat array form as
 * MeshData stores positions (length = vertexCount * 3).
 */
export function meshBoundingRadius(positions: Float32Array | number[]): number {
  const N = positions.length / 3
  if (N === 0) return 1
  const [cx, cy, cz] = meshCentroid(positions)
  let r2 = 0
  for (let i = 0; i < N; i++) {
    const dx = positions[i * 3 + 0] - cx
    const dy = positions[i * 3 + 1] - cy
    const dz = positions[i * 3 + 2] - cz
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 > r2) r2 = d2
  }
  return Math.sqrt(r2)
}
