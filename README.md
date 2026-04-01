## Dynamic Predicate Transfer (RPT+)

This repository contains the implementation of **Dynamic Predicate Transfer (RPT+)**, built on top of [DuckDB v1.3.0](https://github.com/duckdb/duckdb/tree/v1.3-ossivalis). 

Compared to the original Robust Predicate Transfer (RPT), RPT+ introduces several key optimizations. For technical details, please refer to the paper:

> **Robust Predicate Transfer with Dynamic Execution** (PVLDB 2026) · Yiming Qiao, Peter Boncz, Huanchen Zhang [[Link]](https://yimingqiao.github.io/files/rpt_plus.pdf)

Recent DuckDB releases v1.5.0 have further improved execution performance, and some optimizations may overlap with or subsume parts of RPT+, reflecting the continued evolution of modern query execution engines.

## Quick Start

Follow these steps to build RPT+ and run a demonstration of the multi-way join optimization.

### 1. Build & Launch
Compile the project and start the DuckDB shell:

```bash
make release
./build/release/duckdb
```

### 2. Run Example
Run the following SQL to set up data (5M rows) and perform a 3-way join. RPT+ will automatically transfer filters to optimize performance.

```sql
-- 1. Setup Data
CREATE TABLE A AS SELECT i AS id1, i AS id2 FROM range(1, 5000001) AS t(i);
CREATE TABLE B AS SELECT i AS id1, i AS id2 FROM range(1, 5000001) AS t(i);
CREATE TABLE C AS SELECT i AS id1, i AS id2 FROM range(1, 5000001) AS t(i);

-- 2. Execute Query
EXPLAIN ANALYZE        -- Show query execution plan
SELECT count(*)
FROM A JOIN B ON A.id1 = B.id1 JOIN C ON B.id1 = C.id1
WHERE A.id2 % 2 = 0    -- Filter on A
  AND B.id2 % 7 = 0    -- Filter on B
  AND C.id2 % 13 = 0;  -- Filter on C
```

The query plan demonstrates a **chain forward pass** (C → B → A) followed by a **broadcast backward pass** (A → B & C). 

## Advanced Build Configuration

RPT+ follows the same build process as DuckDB. For customized builds:

```bash
make                   # Build optimized release version
make release           # Same as 'make'
make debug             # Build with debug symbols
GEN=ninja make         # Use Ninja as backend
BUILD_BENCHMARK=1 make # Build with benchmark support
```

For more flags, see the [DuckDB Build Configuration Guide](https://duckdb.org/docs/stable/dev/building/build_configuration.html).

### Original RPT Support
To use the **original RPT** implementation on DuckDB 1.3.0, apply the vanilla patch:

```bash
git apply APPLY_ME_TO_GET_VANILLA_RPT.patch
```

## Benchmark

DuckDB includes a built-in implementation of benchmarks. You can build and run them with:

### TPC-H (SF=100)
```bash
BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 CORE_EXTENSIONS='tpch' make -j$(nproc)
build/release/benchmark/benchmark_runner "benchmark/large/tpch-sf100/.*.benchmark" --threads=8
```

### Join Order Benchmark (JOB)
```bash
BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 CORE_EXTENSIONS='tpch' make -j$(nproc)
build/release/benchmark/benchmark_runner "benchmark/imdb/.*.benchmark" --threads=8
```

### Appian Benchmark
```bash
BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 CORE_EXTENSIONS='tpch' make -j$(nproc)
build/release/benchmark/benchmark_runner "benchmark/appian_benchmarks/.*.benchmark" --threads=8
```

### SQLStorm

To run the SQLStorm benchmark:

1. Clone and set up the benchmark framework from the [SQLStorm repository](https://github.com/SQL-Storm/SQLStorm).
2. Download the [StackOverflow Math dataset](https://db.in.tum.de/~schmidt/data/stackoverflow_math.tar.gz) and load it according to SQLStorm’s setup instructions.
3. The list of queries that are executable with DuckDB is available [here](https://github.com/SQL-Storm/SQLStorm/blob/master/v1.0/stackoverflow/valid_queries.csv).

To build a duckdb executable for SQLStorm,

```bash
BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 CXXFLAGS="-DUSE_LOCK_BF=1 -DUSE_SQLSTORM_DP_CONDITION=1" make -j$(nproc)
```

The executable of duckdb is placed at `./build/release/duckdb`.

---
> **Note:**  
> The following section is the *unmodified original README* of DuckDB.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="logo/DuckDB_Logo-horizontal.svg">
    <source media="(prefers-color-scheme: dark)" srcset="logo/DuckDB_Logo-horizontal-dark-mode.svg">
    <img alt="DuckDB logo" src="logo/DuckDB_Logo-horizontal.svg" height="100">
  </picture>
</div>
<br>

<p align="center">
  <a href="https://github.com/duckdb/duckdb/actions"><img src="https://github.com/duckdb/duckdb/actions/workflows/Main.yml/badge.svg?branch=main" alt="Github Actions Badge"></a>
  <a href="https://discord.gg/tcvwpjfnZx"><img src="https://shields.io/discord/909674491309850675" alt="discord" /></a>
  <a href="https://github.com/duckdb/duckdb/releases/"><img src="https://img.shields.io/github/v/release/duckdb/duckdb?color=brightgreen&display_name=tag&logo=duckdb&logoColor=white" alt="Latest Release"></a>
</p>

## DuckDB

DuckDB is a high-performance analytical database system. It is designed to be fast, reliable, portable, and easy to use. DuckDB provides a rich SQL dialect, with support far beyond basic SQL. DuckDB supports arbitrary and nested correlated subqueries, window functions, collations, complex types (arrays, structs, maps), and [several extensions designed to make SQL easier to use](https://duckdb.org/docs/stable/sql/dialect/friendly_sql.html).

DuckDB is available as a [standalone CLI application](https://duckdb.org/docs/stable/clients/cli/overview) and has clients for [Python](https://duckdb.org/docs/stable/clients/python/overview), [R](https://duckdb.org/docs/stable/clients/r), [Java](https://duckdb.org/docs/stable/clients/java), [Wasm](https://duckdb.org/docs/stable/clients/wasm/overview), etc., with deep integrations with packages such as [pandas](https://duckdb.org/docs/guides/python/sql_on_pandas) and [dplyr](https://duckdb.org/docs/stable/clients/r#duckplyr-dplyr-api).

For more information on using DuckDB, please refer to the [DuckDB documentation](https://duckdb.org/docs/stable/).

## Installation

If you want to install DuckDB, please see [our installation page](https://duckdb.org/docs/installation/) for instructions.

## Data Import

For CSV files and Parquet files, data import is as simple as referencing the file in the FROM clause:

```sql
SELECT * FROM 'myfile.csv';
SELECT * FROM 'myfile.parquet';
```

Refer to our [Data Import](https://duckdb.org/docs/stable/data/overview) section for more information.

## SQL Reference

The documentation contains a [SQL introduction and reference](https://duckdb.org/docs/stable/sql/introduction).

## Development

For development, DuckDB requires [CMake](https://cmake.org), Python3 and a `C++11` compliant compiler. Run `make` in the root directory to compile the sources. For development, use `make debug` to build a non-optimized debug version. You should run `make unit` and `make allunit` to verify that your version works properly after making changes. To test performance, you can run `BUILD_BENCHMARK=1 BUILD_TPCH=1 make` and then perform several standard benchmarks from the root directory by executing `./build/release/benchmark/benchmark_runner`. The details of benchmarks are in our [Benchmark Guide](benchmark/README.md).

Please also refer to our [Build Guide](https://duckdb.org/docs/stable/dev/building/overview) and [Contribution Guide](CONTRIBUTING.md).

## Support

See the [Support Options](https://duckdblabs.com/support/) page.
