# Development

## Dev mode

```bash
./scripts/setup.sh    # once
./scripts/dev.sh      # backend with --reload on :8000 + Vite with hot reload on :5173
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend, so the UI and
API behave exactly as in the single-process build.

On Windows, run the two halves in two terminals:

```powershell
cd backend;  .venv\Scripts\python -m uvicorn main:app --reload --port 8000
cd frontend; npm run dev
```

## Project layout

```
backend/
  main.py               FastAPI app: routing + pipeline orchestration; serves frontend/dist
  sessions.py           Session store, persisted to disk (survives restarts)
  obj_io.py             Writes results back in the original polygon/UV layout
  fbx_io.py             FBX via Blender run headless as a subprocess
  blender_scripts/      The scripts Blender runs (shape-key extraction, FBX build)
  infra/                Config (env vars), quotas, compute queue, background jobs, logging
  dt_core/              The math — numpy/scipy only, knows nothing about HTTP
    correspondence.py     Iterative closest-point correspondence (the "wrap" solve)
    transformation.py     Deformation transfer of poses
    surface.py            Closest point on surface, match pruning, projection, anti-fold
    regions.py            Boundary loop → editable / frozen / feathered regions
    smoothing.py          Taubin smoothing
    align.py              Umeyama similarity alignment from markers
    refit.py              Garment refit orchestrator (contact field, presets)
    layers.py             Automatic driven/follower layer classification
    arap.py               ARAP bind (garment keeps its shape)
    collision.py          Push-out / hide-body collision handling
    pins.py, wardrobe.py  Poke pins; proxy wardrobe binding
    solver.py             Sparse solver (MKL Pardiso, SuperLU fallback)
    meshlib/              Mesh loading (vectorized OBJ parser)
  tests/                Smoke tests (standalone scripts)
  tools/                Profiler
frontend/
  src/App.tsx           Layout, wizard, per-tool panels
  src/tools.tsx         Tool metadata + all in-app guide copy
  src/store.ts          Zustand store: session, markers, jobs
  src/api.ts            Typed API client
  src/components/       three.js viewports (MeshViewport) and result preview
  src/changelog.ts      "What's new" entries (bump with package.json version)
demo_assets/            Redistributable sample meshes + the script that prepares them
docs/                   User and developer docs; docs/design/ holds implementation plans
scripts/                setup / start / dev / test launchers
```

## Tests

```bash
./scripts/test.sh            # all backend smoke tests (~2 minutes)
./scripts/test.sh refit wrap # only files whose name contains a pattern
```

Each test in `backend/tests/*_smoke.py` is a standalone script that prints
`[ok]` / `[FAIL]` lines and exits non-zero on failure. Tests whose names
contain `http` (and a few others) talk to a backend on `127.0.0.1:8000`; the
runner starts and stops one on a throwaway workspace, and restarts it with
low mesh caps for `limits_smoke.py`. A single in-process test can also be run
directly: `cd backend && .venv/bin/python tests/region_smoke.py`.

`layer_groundtruth_check.py` is a manual tool, not a test: it compares the
automatic layer classification with a hand-painted mask on your own assets.

Frontend checks:

```bash
cd frontend && npx tsc --noEmit && npm run build
```

CI (`.github/workflows/ci.yml`) runs the backend tests, the frontend build and
a Docker build on every pull request.

## How the pipeline works

**Correspondence (Wrap).** The source is deformed toward the target in a few
iterations. Each iteration solves a sparse least-squares system with three
terms: *smoothness* (neighboring triangles deform alike), *identity* (stay
close to the original shape), and *closest point* (vertices move onto the
target surface), with marker vertices pinned. The closest-point weight ramps
up over the iterations. Wrap is the result of this step; DT Transfer uses it
to build a triangle-to-triangle mapping.

**Wrap robustness (`dt_core/surface.py`).** Matches go to the nearest point on
a target *triangle*, not the nearest vertex, and are pruned:

- **Normal compatibility** (default 60°) — never match through the mesh onto
  the far side or an inner shell.
- **Open-boundary rejection** — matches landing on open borders (eye sockets,
  mouth bags, mesh cuts) are dropped, which removes the classic edge suction.
- **Adaptive distance cut** — matches farther than `max(floor, 3 × median)` are
  dropped, so uncovered areas ride along on smoothness.
- **Anti-fold damping** — triangles that flip during an iteration are pulled
  back toward their previous state.
- **Final projection** — the result is snapped onto the target (alternating
  with smoothing when enabled) and a residual (mean/p95/max) is reported.

**Deformation transfer.** For each pose, the per-triangle deformation of the
source (pose vs. neutral) is applied to the mapped target triangles, and the
target vertices are solved so the result is as consistent with those
deformations as possible.

**Refit** reuses the wrap solver with a per-vertex *conform field* (how strongly
each garment vertex grips the body: seam grip, contact distance, pins, frozen
paint, layer classification), then an ARAP bind pass so the garment keeps its
own shape, then collision handling. See the plans in [design/](design/) for
the reasoning behind each piece.

### Performance

- **MKL Pardiso** (`pypardiso`) replaces SciPy's SuperLU for sparse
  factorizations, 2–4× faster on multi-core x86_64; SuperLU is the automatic
  fallback.
- **Per-session cache** keyed on mesh hashes, markers and parameters: re-running
  with the same settings is near-instant, adding poses only transfers the new
  ones, and changing only the smoothness reuses the source/target setup.
- **Factor reuse** in deformation transfer: the system matrix doesn't depend on
  the pose, so it is factored once and every pose is a back-substitution.
- `cKDTree.query(workers=-1)` for nearest-neighbor queries, vectorized normals
  and OBJ parsing, gzip + an LRU cache for mesh JSON.
- `backend/tools/profile_pipeline.py` prints a per-stage timing breakdown.

The session cache lives in memory per session; intermediate sparse matrices are
not cached to disk (unlike the upstream project) to keep sessions isolated.

## Conventions

- `dt_core` is numpy/scipy only; heavy optional imports happen lazily inside
  functions.
- New behavior ships behind a parameter that defaults to the old behavior, and
  with a smoke test.
- All environment variables are read in `backend/infra/config.py` (or the
  module that owns them) and documented in [CONFIGURATION.md](CONFIGURATION.md).
- User-facing guide text for the tools lives in `frontend/src/tools.tsx`;
  update [TOOLS.md](TOOLS.md) alongside it.
- When releasing, bump `frontend/package.json` and add an entry to
  `frontend/src/changelog.ts`.
