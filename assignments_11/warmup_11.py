# Week 11 Warmup
# Code the Dream, Python for Cloud & AI

from prefect import task
from prefect.logging import get_run_logger


# ---------------------------------------------------------------------------
# Prefect Orchestration
# ---------------------------------------------------------------------------

# Prefect Question 1
#
# A @task is a single unit of work: one API call, one file write, one transform
# step. Prefect tracks its state, can retry it, and captures its logs as its
# own node in the run. A @flow is the orchestrator. It calls tasks in order,
# passes one task's output into the next, and represents the whole run.
#
# For a pure Celsius to Fahrenheit function I would not use @task. It does no
# I/O, has no transient failure modes, and gains nothing from retries. Wrapping
# it only adds state tracking overhead and clutters the UI with a node that can
# never meaningfully fail. I would leave it as a plain function and call it from
# inside whatever task needs the converted value. Tasks earn their keep when
# something can fail or is worth observing on its own.


# Prefect Question 2
#
# Decorator line only:
#
@task(retries=3, retry_delay_seconds=30)


# Prefect Question 3
#
# I would open the failed flow run from the run list, then click the transform
# task node (the red, Failed one). Its Logs tab is where the detail lives. I
# expect the exception traceback there: the exception type and message, the line
# in transform that raised it, and the timestamp. The flow view also confirms
# load was never scheduled, since a failed upstream task stops the tasks that
# depend on it.


# ---------------------------------------------------------------------------
# Production Patterns
# ---------------------------------------------------------------------------

# Production Question 1
#
# raise_for_status() inspects the response code and raises an HTTPError for any
# 4xx or 5xx response, and does nothing on a 2xx. It is better than
# `if response.status_code != 200: print("error")` because a print stops
# nothing. The task keeps running, returns a bad payload, and Prefect still
# marks it Completed, so the bad data flows downstream silently.
#
# On a 500 error:
#   With raise_for_status(): the task raises, Prefect marks it Failed, the
#   downstream transform and load tasks never run, and nothing bad is written.

#   With the print check: the task prints "error" but keeps going, returns the
#   500 body, and transform and load run on garbage. The failure is invisible
#   until someone notices the output is wrong.


# Production Question 2
#
# overwrite=True lets the load step replace an existing blob at that path rather
# than erroring. Without it, upload_blob raises ResourceExistsError when a blob
# already lives at final/{today}/weather_etl.json, so re-running the same day
# would fail at load.
#
# One precise note on this exact scenario: the crash was in transform, so load
# never ran on the first attempt and no blob was written. The path is therefore
# empty when I re-run, and overwrite=True is not actually exercised by this
# particular re-run. Where it does protect me is the general case: any time a
# date already produced output (a prior successful run, or a later debugging
# pass), overwrite=True keeps the load idempotent instead of failing on a
# duplicate blob.


# Production Question 3

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")