"""
Times the UDF, native and hex string versions of every natively implemented h3spark function. Not run in CI.

    uv run python bench/bench.py --rows 1000000 --cores 8
"""

import argparse
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import h3spark
import h3spark.hex as h3hex
from h3spark import REGISTRY

RES = 9
# Non-cell arguments by parameter name. Cells are at RES, so RES is a valid parent and child resolution
ARGS = {
    "h3_str": "cell_str",
    "res": RES,
    "resolution": RES,
    "child_resolution": RES + 1,
    "res_parent": RES - 4,
    "res_child": RES + 2,
    "child_pos": 30,
}


def best_time(df, column, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        df.select(column).write.format("noop").mode("overwrite").save()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.master(f"local[{args.cores}]")
        .config("spark.ui.enabled", False)
        .config("spark.ui.showConsoleProgress", False)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    lat = F.degrees(F.asin(F.rand(1) * 2 - 1))
    lng = F.rand(2) * 360 - 180
    df = (
        spark.range(args.rows)
        .select(h3spark.latlng_to_cell(lat, lng, RES).alias("cell"))
        .withColumn("cell_str", h3spark.int_to_str("cell"))
        .cache()
    )
    df.count()

    print(f"{args.rows:,} rows, local[{args.cores}], best of {args.repeat} (seconds)")
    print(f"{'function':<30}{'udf':>8}{'native':>8}{'hex':>8}")
    for name, entry in REGISTRY.items():
        if not entry.native:
            continue
        params = [p for p in entry.signature.parameters if p not in entry.static]
        long_args = [ARGS.get(p, "cell") for p in params]
        hex_args = ["cell_str" if p in entry.cells else ARGS.get(p) for p in params]
        variants = [entry.udf(*long_args), entry.native(*long_args)]
        if entry.hex:
            variants.append(getattr(h3hex, name)(*hex_args))
        times = "".join(f"{best_time(df, v, args.repeat):8.2f}" for v in variants)
        print(f"{name:<30}{times}")


if __name__ == "__main__":
    main()
