"""
Shared test helper for the async job model.

/run and /wrap no longer block until the compute finishes — they return a
`{job_id}` immediately and the work happens on a background thread. Tests
submit the job, then poll GET /sessions/{id}/job until it reaches a terminal
state, and read the endpoint payload from job["result"].

`http` is the per-test HTTP helper (same signature as in the smoke tests:
http(method, path, body=None, ...) -> parsed JSON).
"""
import time


def wait_for_job(http, session_id, timeout=600.0, interval=0.25):
    """Poll GET /job until done/error. Returns the result payload dict on
    success; raises RuntimeError on job error or TimeoutError if it never
    finishes."""
    deadline = time.time() + timeout
    while True:
        job = http("GET", f"/api/sessions/{session_id}/job")
        state = job.get("state")
        if state == "done":
            return job.get("result") or {}
        if state == "error":
            raise RuntimeError(f"job failed: {job.get('error')}")
        if time.time() > deadline:
            raise TimeoutError(f"job did not finish in {timeout}s (last state={state})")
        time.sleep(interval)


def submit_and_wait(http, session_id, kind, payload, **kw):
    """POST /run or /wrap (kind in {'run','wrap'}), then block until it
    finishes. Returns the result payload."""
    resp = http("POST", f"/api/sessions/{session_id}/{kind}", payload)
    if not isinstance(resp, dict) or "job_id" not in resp:
        raise AssertionError(f"expected a job_id from /{kind}, got {resp!r}")
    return wait_for_job(http, session_id, **kw)
