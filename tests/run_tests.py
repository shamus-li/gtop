#!/usr/bin/env python3
"""
Test runner for gtop tests using pytest
"""

import subprocess
import sys
import os

def main():
    """Run all tests using pytest"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)

    print("=" * 60)
    print("🧪 RUNNING GTOP TESTS WITH PYTEST")
    print("=" * 60)

    # Change to project root directory for proper imports
    os.chdir(project_root)

    # Run pytest with verbose output and coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ]

    try:
        result = subprocess.run(cmd, timeout=60)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())