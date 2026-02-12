"""Configuration for pytest to support dual-backend testing."""

import os
import subprocess
import sys

import pytest


def pytest_addoption(parser):
    """Add custom pytest option for backend selection."""
    parser.addoption(
        "--backend",
        action="store",
        default="both",
        help="Backend to use: auto, numpy, opt_einsum, or both (default: both)",
    )


def pytest_configure(config):
    """Configure based on backend option."""
    backend = config.getoption("--backend")
    
    if backend == "both":
        # Run with both backends by invoking pytest twice
        print("\n[conftest] Running tests with both backends...")
        
        # Get the command line args, excluding --backend
        args = [arg for arg in sys.argv[1:] if not arg.startswith("--backend")]
        
        # Run with numpy
        print("\n" + "="*50)
        print("RUNNING WITH NUMPY BACKEND")
        print("="*50)
        numpy_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--backend=numpy"] + args,
            cwd="."
        )
        
        # Run with opt_einsum
        print("\n" + "="*50)
        print("RUNNING WITH OPT_EINSUM BACKEND")
        print("="*50)
        opt_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--backend=opt_einsum"] + args,
            cwd="."
        )
        
        # Exit cleanly with combined status
        exit_code = max(numpy_result.returncode, opt_result.returncode)
        os._exit(exit_code)


@pytest.fixture(scope="session", autouse=True)
def setup_backend(request):
    """Configure backend based on pytest option."""
    backend = request.config.getoption("--backend")
    
    # Skip if we're in the parent process of --backend=both
    if backend == "both":
        pytest.skip("Running in subprocess mode")
    
    if backend == "numpy":
        from eincraft.utils import disable_opt_einsum
        disable_opt_einsum()
        print("\n[conftest] Using numpy backend")
    elif backend == "opt_einsum":
        from eincraft.utils import oe
        if oe is None:
            pytest.skip("opt_einsum not installed")
        print("\n[conftest] Using opt_einsum backend")
    elif backend == "auto":
        from eincraft.utils import oe
        if oe is not None:
            print("\n[conftest] Using auto-detected backend (opt_einsum)")
        else:
            print("\n[conftest] Using auto-detected backend (numpy)")
