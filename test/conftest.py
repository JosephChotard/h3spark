import os
import sys

import pytest
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.config("spark.sql.shuffle.partitions", 4)
        .config("spark.ui.enabled", False)
        .getOrCreate()
    )
    yield spark
    spark.stop()
