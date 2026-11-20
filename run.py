"""Single-command launcher for the Food Photo to Recipe Finder app (R7)."""

from __future__ import annotations

import uvicorn

from download_weights import ensure_weights


def main() -> None:
    print("============================================================", flush=True)
    print(" Starting Food Photo to Recipe Finder (FastAPI + ONNX INT8) ", flush=True)
    print(" URL:      http://127.0.0.1:8000                            ", flush=True)
    print(" API Docs: http://127.0.0.1:8000/docs                       ", flush=True)
    print("============================================================", flush=True)
    ensure_weights()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
