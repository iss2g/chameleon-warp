# Chameleon Warp — single image: the built web UI + the FastAPI backend in one
# process on port 8000. See docs/INSTALL.md.
#
#   docker compose up -d --build          # recommended
#   docker build -t chameleon-warp .      # plain docker
#
# Build args:
#   WITH_BLENDER=1 (default) bundles Blender for FBX import/export (~350 MB,
#                  x86_64 only — skipped automatically on arm64).
#   WITH_BLENDER=0 gives a smaller image that handles .obj only.

# ---- 1. Frontend build ------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Runtime -------------------------------------------------------------
# Python 3.9 is required: the algorithm core pins numpy 1.20 / scipy 1.6.
FROM python:3.9-slim-bookworm

ARG WITH_BLENDER=1
ARG BLENDER_VERSION=4.2.0
ARG TARGETARCH

# Blender needs a handful of X/GL client libraries even in --background mode.
RUN set -eux; \
    if [ "$WITH_BLENDER" = "1" ] && [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            curl xz-utils ca-certificates \
            libegl1 libgl1 libgomp1 libsm6 libx11-6 libxext6 libxfixes3 \
            libxi6 libxkbcommon0 libxrender1 libxxf86vm1; \
        series="$(echo "$BLENDER_VERSION" | cut -d. -f1,2)"; \
        curl -fsSL "https://download.blender.org/release/Blender${series}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
            -o /tmp/blender.tar.xz; \
        mkdir -p /opt/blender; \
        tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1; \
        rm /tmp/blender.tar.xz; \
        ln -s /opt/blender/blender /usr/local/bin/blender; \
        apt-get purge -y --auto-remove curl xz-utils; \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Skipping Blender (WITH_BLENDER=$WITH_BLENDER, arch=${TARGETARCH:-?}) — FBX disabled"; \
    fi

WORKDIR /app/backend
COPY backend/requirements.txt ./
# pypardiso (Intel MKL) is a best-effort speedup: x86_64 only, and the solver
# falls back to SciPy's SuperLU without it.
RUN set -eux; \
    grep -viE '^\s*pypardiso' requirements.txt > /tmp/req-core.txt; \
    pip install --no-cache-dir -r /tmp/req-core.txt; \
    pip install --no-cache-dir "$(grep -iE '^\s*pypardiso' requirements.txt)" \
        || echo "pypardiso unavailable on this platform — using SuperLU"; \
    rm /tmp/req-core.txt

COPY backend/ ./
COPY --from=frontend /src/frontend/dist /app/frontend/dist

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/workspace \
    && chown -R app:app /data
USER app

ENV DT_WORKSPACE_ROOT=/data/workspace \
    TQDM_DISABLE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
