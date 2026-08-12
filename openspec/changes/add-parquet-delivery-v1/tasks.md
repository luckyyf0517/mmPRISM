## 1. Contract And Dependency

- [x] 1.1 Add the `data-parquet` optional dependency and lock it.
- [x] 1.2 Define delivery metadata, part inventory and index contract with lazy PyArrow loading.
- [x] 1.3 Update data-delivery Authority with copied input binding and row-group policy.

## 2. Readers And Materializer

- [x] 2.1 Implement typed pose reconstruction and translation Parquet readers.
- [x] 2.2 Implement deterministic capacity plan and atomic materializer from frozen JSONL manifest/split.
- [x] 2.3 Implement complete inventory/index/checksum/row validation.

## 3. Verification And Handoff

- [x] 3.1 Add tiny source-adapter to Parquet reader parity fixtures for both products.
- [x] 3.2 Cover split isolation, 1,024-row/64-part placement, no-clobber, tamper and missing optional dependency behavior.
- [x] 3.3 Run targeted tests, static checks, documentation audit and record workspace status.
