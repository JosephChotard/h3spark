import contextlib
import io
import os
import sys

import pytest
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Run the tests through a local Spark Connect server instead of a classic session (needs pyspark[connect] >= 4)
REMOTE = os.environ.get("H3SPARK_TEST_REMOTE") == "1"


@pytest.fixture(scope="session")
def spark():
    builder = SparkSession.builder
    if REMOTE:
        builder = builder.remote("local[*]")
    spark = (
        builder.config("spark.sql.shuffle.partitions", 4)
        # Fail instead of silently falling back to (much slower) interpreted evaluation, e.g. if a native expression
        # grows beyond the JVM's method size limit
        .config("spark.sql.codegen.fallback", False)
        .config("spark.ui.enabled", False)
        .getOrCreate()
    )
    yield spark
    spark.stop()


def physical_plan(df) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        df.explain()
    return out.getvalue()
