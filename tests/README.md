# gtop Test Suite

This directory contains comprehensive tests for the gtop GPU parsing functionality.

## Test Files

### `test_gpu_parsing.py`
Unit tests for core GPU parsing functions:
- Regular GPU parsing (non-sharded)
- Sharded GPU detection and parsing
- Job TRES allocation parsing
- Edge cases and error handling

### `test_integration.py`
Integration tests using real cluster data:
- Tests against actual sinfo and gtop command output
- Validates parsing of complex real-world GRES strings
- Covers edge cases found in production data

### `run_tests.py`
Test runner that executes all tests using pytest and provides detailed results.

## Sample Data Files

### `sinfo-output.txt`
Real output from `sinfo -O nodehost:100,gres:100,gresused:100,cpusstate:100,allocmem:100,memory:100 -h` command showing various GPU configurations including:
- Regular GPUs
- Sharded GPUs (with and without separate shard entries)
- Mixed GPU types
- CPU-only nodes

### `gtop-output.txt`
Real output from `sacct` command showing job allocations with TRES strings including CPU, memory, and GPU allocations.

## Running Tests

```bash
# Run all tests with pytest
python tests/run_tests.py

# Or run pytest directly
pytest tests/ -v

# Run individual test files
pytest tests/test_gpu_parsing.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=gtop --cov-report=term-missing
```

## Test Coverage

The tests validate:

1. **GPU Type Detection**:
   - ✅ Regular GPUs are not marked as sharded
   - ✅ True sharded GPUs are properly identified
   - ✅ Mixed GPU types are supported

2. **Sharding Logic**:
   - ✅ Simple shard ranges like `(S:0)` are treated as regular GPUs
   - ✅ Complex ranges like `(S:0-1)` are detected as shared
   - ✅ Comma patterns like `(S:0,3)` are treated as separate allocations
   - ✅ Nodes with `shard:` entries are properly handled

3. **Usage Parsing**:
   - ✅ GPU double-counting is prevented (`gres/gpu:type=N` and `gres/gpu=N`)
   - ✅ CPU-only jobs are handled correctly
   - ✅ Multi-GPU allocations are parsed accurately

4. **Real Data Compatibility**:
   - ✅ Works with actual SLURM cluster configurations
   - ✅ Handles edge cases found in production
   - ✅ Maintains backward compatibility

## Adding New Tests

When adding new test cases:
1. Add unit tests to `test_gpu_parsing.py` for new parsing logic
2. Add integration tests to `test_integration.py` for real-world scenarios
3. Update this README with coverage information
4. Ensure tests pass with `pytest tests/ -v`
