# Contributing

Bug reports, ideas and pull requests are welcome.

**Reporting a bug:** open an issue with what you did, what you expected and
what happened. Mesh-related problems are much easier to fix with the meshes
(or a small mesh that reproduces it) and the exported markers JSON. Include
the backend log (`docker compose logs` or the terminal running
`start.sh`) and your platform.

**Pull requests:**

1. Set up the dev environment: `./scripts/setup.sh`, then `./scripts/dev.sh`
   (see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)).
2. Keep changes focused; new solver behavior goes behind a parameter that
   defaults to the old behavior.
3. Add or update a smoke test in `backend/tests/` and run `./scripts/test.sh`.
4. For frontend changes: `cd frontend && npx tsc --noEmit && npm run build`.
5. Update the docs in `docs/` (and the in-app guide in `frontend/src/tools.tsx`)
   if you change how a tool behaves.

By contributing you agree that your contribution is licensed under the
project's [MIT License](LICENSE).
