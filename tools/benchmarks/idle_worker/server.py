"""Local worker HTTP server with observational batch tracing for the benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import uvicorn
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.batcher import FormattedBatch, HasCost
from sie_server.core.worker import ModelWorker, model_worker
from sie_server.core.worker.types import RequestMetadata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--models-dir", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--port", type=int, default=8373)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
trace_stream = args.trace.open("w")
original_process_batch = ModelWorker._process_batch


async def traced_process_batch(
    self: ModelWorker, batch: FormattedBatch[HasCost, RequestMetadata], *, engine_queue_owned: bool = False
) -> None:
    await original_process_batch(self, batch, engine_queue_owned=engine_queue_owned)
    ids = sorted({str(item.id) for metadata in batch.metadata for item in metadata.items})
    trace_stream.write(json.dumps({"size": batch.size, "request_ids": ids}) + "\n")
    trace_stream.flush()


ModelWorker._process_batch = traced_process_batch  # ty: ignore[invalid-assignment]
source = Path(model_worker.__file__)
print(
    json.dumps({"worker_path": str(source), "worker_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}),
    flush=True,
)
app = AppFactory.create_app(
    AppStateConfig(
        models_dir=args.models_dir,
        device=args.device,
        model_filter=[args.model],
        preload_models=[args.model],
        remote_serving=False,
    )
)
uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False, loop="asyncio")
