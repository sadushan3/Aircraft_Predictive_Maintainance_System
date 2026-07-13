"""
General utilities for CA-EDT-AHMA.
"""

from __future__ import annotations

print("[PROGRESS] Loaded Backend/app/utils/Anomaly_Health_Monitering/utils.py")
import gc
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Optional

import os as _os
import sys as _sys

if __package__ in {None, ""}:
    _backend_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
    if _backend_root not in _sys.path:
        _sys.path.insert(0, _backend_root)

from app.config.Anomaly_Health_Monitering.config import Config
from app.utils.Anomaly_Health_Monitering.file_utils import atomic_write_json
from app.utils.Anomaly_Health_Monitering.logging_utils import get_logger

logger = get_logger(__name__)


def collect_memory() -> None:
    """
    Release cyclic garbage between memory-heavy pipeline stages.
    """
    gc.collect()


@dataclass
class StageResult:
    """
    Standard stage result.

    Attributes:
        stage_name: Name of the pipeline stage.
        status: success or failed.
        message: Human-readable message.
        output_file: Optional output file path.
        records_count: Optional record count.
        elapsed_seconds: Runtime.
    """

    stage_name: str
    status: str
    message: str
    output_file: Optional[str]
    records_count: Optional[int]
    elapsed_seconds: float


def run_stage_safely(
    stage_name: str,
    stage_function: Callable[[], Dict[str, object]],
) -> StageResult:
    """
    Run one pipeline stage safely.

    If the current stage fails, previously executed outputs remain unchanged.

    Args:
        stage_name: Name of the stage.
        stage_function: Function to execute.

    Returns:
        StageResult: Stage result.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/utils.py::run_stage_safely")
    started = perf_counter()

    try:
        logger.info("Starting stage: %s", stage_name)
        result = stage_function()
        elapsed = round(perf_counter() - started, 3)

        stage_result = StageResult(
            stage_name=stage_name,
            status=str(result.get("status", "success")),
            message=str(result.get("message", f"{stage_name} completed.")),
            output_file=str(result.get("output_file")) if result.get("output_file") else None,
            records_count=(
                int(result["records_count"])
                if result.get("records_count") is not None
                else None
            ),
            elapsed_seconds=elapsed,
        )

        atomic_write_json(
            {
                "last_stage": stage_result.stage_name,
                "last_status": stage_result.status,
                "message": stage_result.message,
                "output_file": stage_result.output_file,
                "records_count": stage_result.records_count,
                "elapsed_seconds": stage_result.elapsed_seconds,
            },
            Config.REPORT_DIR / "stage_manifest.json",
        )

        logger.info("Stage completed successfully: %s", stage_name)
        return stage_result

    except Exception as exc:
        elapsed = round(perf_counter() - started, 3)

        logger.exception("Stage failed safely: %s", stage_name)

        stage_result = StageResult(
            stage_name=stage_name,
            status="failed",
            message=f"{stage_name} failed: {exc}",
            output_file=None,
            records_count=None,
            elapsed_seconds=elapsed,
        )

        atomic_write_json(
            {
                "last_stage": stage_result.stage_name,
                "last_status": stage_result.status,
                "message": stage_result.message,
                "output_file": stage_result.output_file,
                "records_count": stage_result.records_count,
                "elapsed_seconds": stage_result.elapsed_seconds,
                "error_trace": traceback.format_exc(),
                "safety_note": (
                    "Previous successfully generated files were not deleted."
                ),
            },
            Config.REPORT_DIR / "stage_manifest.json",
        )

        return stage_result

    finally:
        collect_memory()


def response_dict(
    status: str,
    message: str,
    output_file: Optional[str] = None,
    records_count: Optional[int] = None,
    data: Optional[object] = None,
) -> Dict[str, object]:
    """
    Create a standard response dictionary.

    Args:
        status: Response status.
        message: Response message.
        output_file: Optional output path.
        records_count: Optional record count.
        data: Optional data payload.

    Returns:
        Dict[str, object]: Response dictionary.
    """
    print("[PROGRESS] Entering Backend/app/utils/Anomaly_Health_Monitering/utils.py::response_dict")
    response: Dict[str, object] = {
        "status": status,
        "message": message,
    }

    if output_file is not None:
        response["output_file"] = output_file

    if records_count is not None:
        response["records_count"] = records_count

    if data is not None:
        response["data"] = data

    return response


if __name__ == "__main__":
    logger.info("General utilities loaded.")
    print(response_dict(status="success", message="Utilities are valid."))
