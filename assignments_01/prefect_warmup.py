# --- Pipelines ---

# Pipeline Q2
import numpy as np
import pandas as pd
from prefect import task, flow


arr_pipeline = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "mode":   series.mode()[0]
    }


@flow
def pipeline_flow():
    s = create_series(arr_pipeline)
    s = clean_data(s)
    result = summarize_data(s)
    for key, val in result.items():
        print(f"{key}: {val:.4f}")
    return result


if __name__ == "__main__":
    pipeline_flow()


# --- Reflection ---

# Q: This pipeline is simple -- just three small functions on a handful of numbers.
#    Why might Prefect be more overhead than it is worth here?
'''
For what is basically three function calls on 12 numbers, Prefect adds a lot of boilerplate and runtime overhead. 
You need to install the library, add decorations to each function, and start The task runner tools of Prefect 
just to get the same result as if you called the functions directly. The observability features (dashboard, logs, retries) 
don't add much value when the pipeline runs in less than a millisecond and doesn't connect to any outside systems. 
For something this small, writing it in plain Python is faster, easier to read, and easier to fix.
'''

# Q: Describe some realistic scenarios where a framework like Prefect could still be
#    useful, even if the pipeline logic itself stays simple like in this case.
#
'''
Prefect is still the best choice when the pipeline needs to run reliably in production, even with simple logic. Some examples are:
- Scheduled runs: If this pipeline pulls new data every night and puts the results into a database, Prefect takes care of 
scheduling, retrying on failure, and alerting without you having to do any of that work yourself.
- External I/O: When a step reads from an S3 bucket, calls an API, or writes to a database, temporary failures become a big problem. 
With Prefect's retry logic and failure visibility, those steps are much safer to use.
- Team environments: When more than one person runs or watches pipelines, the Prefect dashboard shows everyone what ran, when, 
and if it worked. A plain script can't do that.
- Growing pipelines: a pipeline that starts out simple often gets bigger. When you start with Prefect, you already have a structure 
in place for adding branching logic, parallel tasks, or alerts later on.
'''