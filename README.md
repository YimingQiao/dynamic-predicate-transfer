## Dynamic Predicate Transfer

This repository contains the implementation of **Dynamic Predicate Transfer (RPT+)**, built on top of [DuckDB v1.3.0](https://github.com/duckdb/duckdb/tree/v1.3-ossivalis). It provides a customized version of DuckDB. Compared to the original Robust Predicate Transfer (RPT), RPT+ introduces the following key improvements:

- An **asymmetric filter transfer plan** to reduce redundant Bloom Filter (BF) construction.
- A **cascade filter mechanism** that combines min-max and Bloom filters for hierarchical filtering efficiency.
- A **dynamic pipeline strategy** that adapts filter creation and transfer based on runtime selectivity.

## Build

You can build this repository in the same way as the original DuckDB. A `Makefile` wraps the build process. For available build targets and configuration flags, see the [DuckDB Build Configuration Guide](https://duckdb.org/docs/stable/dev/building/build_configuration.html).

```bash
make                   # Build optimized release version
make release           # Same as 'make'
make debug             # Build with debug symbols
GEN=ninja make         # Use Ninja as backend
BUILD_BENCHMARK=1 make # Build with benchmark support
```

## Baselines

- **RPT (Robust Predicate Transfer)**: [https://github.com/embryo-labs/Robust-Predicate-Transfer](https://github.com/embryo-labs/Robust-Predicate-Transfer)
- **DuckDB v1.3.0**: [https://github.com/duckdb/duckdb/tree/v1.3-ossivalis](https://github.com/duckdb/duckdb/tree/v1.3-ossivalis)

## Benchmark

### Join Order Benchmark (JOB)

DuckDB includes a built-in implementation of the Join Order Benchmark. You can build and run it with:

```bash
BUILD_BENCHMARK=1 BUILD_TPCH=1 BUILD_TPCDS=1 BUILD_HTTPFS=1 CORE_EXTENSIONS='tpch' make
build/release/benchmark/benchmark_runner "benchmark/imdb/.*.benchmark" --threads=1
```

### SQLStorm

To run the SQLStorm benchmark:

1. Clone and set up the benchmark framework from the [SQLStorm repository](https://github.com/SQL-Storm/SQLStorm).
2. Download the [StackOverflow Math dataset](https://db.in.tum.de/~schmidt/data/stackoverflow_math.tar.gz) and load it according to SQLStorm’s setup instructions.
3. The list of queries that are executable with DuckDB is available [here](https://github.com/SQL-Storm/SQLStorm/blob/master/v1.0/stackoverflow/valid_queries.csv).


> Below is the original DuckDB's README.


---

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
