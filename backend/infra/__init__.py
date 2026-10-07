"""
Serving concerns that sit between HTTP and the math.

Config knobs, disk quotas, the compute queue, background jobs, logging and
upload serialization live here. The math-heavy modules in `dt_core` know
nothing about HTTP, sessions, or quotas — they just consume meshes and
produce meshes, the same way they'd behave in a CLI tool.
"""
