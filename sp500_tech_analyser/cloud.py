import logging

from flask import Flask, jsonify, request as flask_request

from .config import AppConfig
from .logging_utils import configure_logging
from .pipeline import capture_latest_snapshot

logger = logging.getLogger(__name__)


def parse_and_store(request):
    configure_logging()
    config = AppConfig.from_env()
    logger.info("Received cloud capture request")
    result = capture_latest_snapshot(config=config, upload_to_gcs=True)
    return jsonify(
        {
            "provider": result["provider"],
            "snapshot_at": result["snapshot_at"],
            "gcs_blob": result["gcs_blob"],
            "local_path": result["local_path"],
            "payload": result["payload"],
        }
    )


def run_local_server() -> None:
    configure_logging()
    logger.info("Starting local cloud-capture server on http://127.0.0.1:8080")
    app = Flask(__name__)

    def local_handler():
        return parse_and_store(flask_request)

    app.add_url_rule("/", "parse_and_store", local_handler, methods=["GET", "POST"])
    app.run(host="127.0.0.1", port=8080, debug=True)
