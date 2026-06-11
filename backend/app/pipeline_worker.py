from __future__ import annotations

import json
import sys

from .main import ProcessRequest, process_document


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python -m app.pipeline_worker <document_id> <job_id> <request_json>", file=sys.stderr)
        return 2

    document_id = sys.argv[1]
    job_id = sys.argv[2]
    request_payload = json.loads(sys.argv[3])
    process_document(document_id, job_id, ProcessRequest(**request_payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
