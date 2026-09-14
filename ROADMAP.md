# Roadmap

## Available

- Materialized and streaming queries
- Prepared statements and batched execution
- Arrow, array, and DataFrame output
- Scalar, Arrow, aggregate, and table UDFs
- Arrow registration and appender ingestion
- Async execution, cancellation, progress, and profiling
- Dataset transforms, splits, and backend materialization
- Free-threaded CPython support

## Planned

- Infer schemas for Arrow sources with no record batches.
- Enable streaming automatically for batch iterators.
- Load optional DuckDB extensions at runtime.
- Add free-threaded CI and ThreadSanitizer coverage.
- Validate array backends under free-threaded Python.
- Add group-aware and stratified dataset splits.
- Add categorical encoding and streaming datasets.
