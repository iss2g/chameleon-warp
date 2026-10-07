"""
No-op cache stubs. The original project caches intermediate sparse matrices on disk
to speed up repeated runs with the same meshes. For the interactive UI we run each
job on user-provided meshes with arbitrary marker sets, so disk caching only causes
cross-session pollution. Same interface, but get() always returns None and store()
is a no-op.
"""
from dataclasses import dataclass
from typing import Tuple, Callable, Optional


class SparseMatrixCache:
    def __init__(self, suffix: str = "", prefix: str = "", path: str = ".cache"):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "SparseMatrixCache"
        hashid: str
        shape: Tuple[int, ...]

        def get(self):
            return None

        def store(self, data):
            return None

        def cache(self, func: Callable):
            return func()

    def entry(self, hashid: str, shape: Tuple[int, ...]):
        return SparseMatrixCache.Entry(self, hashid, shape)


class CorrespondenceCache:
    def __init__(self, suffix: str = "", prefix: str = "", path: str = ".cache"):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "CorrespondenceCache"
        hashid: str

        def get(self):
            return None

        def store(self, data):
            return None

        def cache(self, func: Callable, *args, **kwargs):
            return func(*args, **kwargs)

    def entry(self, hashid: str):
        return CorrespondenceCache.Entry(self, hashid)


class DeformedMeshCache:
    def __init__(self, suffix: str = "", prefix: str = "", path: str = ".cache"):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "DeformedMeshCache"
        hashid: str
        original: Optional["object"] = None

        def get(self):
            return None

        def store(self, data):
            return None

        def cache(self, func: Callable):
            return func()

    def entry(self, original, salts=()):
        return DeformedMeshCache.Entry(self, "noop", original)
