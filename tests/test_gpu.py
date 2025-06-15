# this_file: tests/test_gpu.py

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

# Import the module to be tested
from imgcolorshine import gpu as gpu_module


# Reset global flags before each test to ensure isolation
@pytest.fixture(autouse=True)
def reset_globals():
    """"""
    gpu_module.GPU_AVAILABLE = False
    gpu_module.CUPY_AVAILABLE = False
    gpu_module.JAX_AVAILABLE = False
    gpu_module._jax_checked = False  # Reset JAX check flag

    # Ensure that if cupy or jax were mocked, they are unmocked or re-mocked cleanly
    if "cupy" in sys.modules:
        del sys.modules["cupy"]
    if "jax" in sys.modules:
        del sys.modules["jax"]
    if "jax.numpy" in sys.modules:
        del sys.modules["jax.numpy"]

    # Reload the gpu_module to re-evaluate initial imports and flags based on mocks
    # This is crucial if mocks are applied globally or affect module-level try-except blocks
    importlib.reload(gpu_module)


# --- Mocking Helpers ---
class MockCuPy:
    """"""

    class cuda:
        """"""

        class Device:
            """"""

            def __init__(self, device_id=0):  # Added device_id to match potential usage
                """"""
                self.name = "MockCuPyDevice"
                self.mem_info = (
                    1024 * 1024 * 1000,
                    1024 * 1024 * 2000,
                )  # 1GB free, 2GB total

            def synchronize(self):  # Add synchronize if it's called
                """"""

        class runtime:
            """"""

            @staticmethod
            def runtimeGetVersion():
                return 11020  # Mock CUDA version

        @staticmethod
        def is_available():
            return True  # Default to available for mock

        class MemoryPool:
            """"""

            def __init__(self):
                """"""
                self.used_bytes_val = 0
                self.total_bytes_val = 0
                self.n_free_blocks_val = 0

            def malloc(self, size):
                """"""
                return mock.MagicMock()  # Mock allocation

            def free_all_blocks(self):
                """"""

            def used_bytes(self):
                return self.used_bytes_val

            def total_bytes(self):
                return self.total_bytes_val

            def n_free_blocks(self):
                return self.n_free_blocks_val

        @staticmethod
        def set_allocator(allocator):
            pass

    @staticmethod
    def asarray(x):
        return np.asarray(x)  # For simplicity, mock CuPy arrays as NumPy arrays

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)

    @staticmethod
    def array(x, dtype=None):  # Add if cp.array is used
        return np.array(x, dtype=dtype)


class MockJax:
    """"""

    class numpy:
        """"""

        @staticmethod
        def asarray(x):
            return np.asarray(x)  # Mock JAX arrays as NumPy arrays

        # Add other jnp functions if they are used by ArrayModule.xp
        # For example, if arithmetic operations from self.xp are used directly
        add = staticmethod(np.add)
        subtract = staticmethod(np.subtract)
        # ... etc.

    @staticmethod
    def devices(device_type=None):
        if device_type == "gpu":
            return ["MockJaxGPU"]
        # If JAX is active and GPU is globally available (mocked), return MockJaxGPU.
        # This makes the mock consistent with a scenario where _check_jax_available found a GPU.
        if gpu_module.JAX_AVAILABLE and gpu_module.GPU_AVAILABLE:
            return ["MockJaxGPU"]
        return ["MockJaxCPU"]


# --- ArrayModule Tests ---
def test_array_module_cpu_fallback():
    """"""
    # Ensure no GPU libs are found (default state of reset_globals often)
    with mock.patch.dict(sys.modules, {"cupy": None, "jax": None, "jax.numpy": None}):
        importlib.reload(gpu_module)  # Reload to reflect absence of modules
        am = gpu_module.ArrayModule(backend="auto")
        assert am.backend == "cpu"
        assert am.xp is np
        test_array = np.array([1, 2, 3])
        assert am.to_device(test_array) is test_array  # Should be same object for numpy
        assert am.to_cpu(test_array) is test_array


@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.GPU_AVAILABLE", True)
def test_array_module_cupy_auto():
    """"""
    # Mock CuPy as available
    sys.modules["cupy"] = MockCuPy  # Assign the class, not an instance
    gpu_module.CUPY_AVAILABLE = True
    gpu_module.GPU_AVAILABLE = True

    am = gpu_module.ArrayModule(backend="auto")
    assert am.backend == "cupy"
    assert am.xp == MockCuPy

    test_array = np.array([1, 2, 3])
    # In our mock, asarray returns a numpy array, so we check type or a special attribute if needed
    device_array = am.to_device(test_array)
    assert isinstance(device_array, np.ndarray)  # MockCuPy.asarray returns np.ndarray

    cpu_array = am.to_cpu(device_array)
    assert isinstance(cpu_array, np.ndarray)


@mock.patch("imgcolorshine.gpu._check_jax_available", return_value=True)
@mock.patch("imgcolorshine.gpu.JAX_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.GPU_AVAILABLE", True)
def test_array_module_jax_auto_if_cupy_not_present(mock_check_jax):
    """"""
    # Mock JAX as available, CuPy as not
    with mock.patch.dict(sys.modules, {"cupy": None}):  # Ensure CuPy is not found
        gpu_module.CUPY_AVAILABLE = False  # Explicitly set CuPy not available
        gpu_module.JAX_AVAILABLE = True  # Set by mock
        gpu_module.GPU_AVAILABLE = True  # Set by mock
        sys.modules["jax"] = MockJax()
        sys.modules["jax.numpy"] = MockJax.numpy

        am = gpu_module.ArrayModule(backend="auto")
        assert am.backend == "jax"
        assert am.xp == MockJax.numpy

        test_array = np.array([1, 2, 3])
        device_array = am.to_device(test_array)
        assert isinstance(
            device_array, np.ndarray
        )  # MockJax.numpy.asarray returns np.ndarray

        cpu_array = am.to_cpu(device_array)
        assert isinstance(cpu_array, np.ndarray)


def test_array_module_force_cpu():
    """"""
    am = gpu_module.ArrayModule(backend="cpu")
    assert am.backend == "cpu"
    assert am.xp is np


@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
def test_array_module_force_cupy_unavailable_fallback():
    """"""
    # Simulate CuPy import successful but is_available() is false or runtime error
    gpu_module.CUPY_AVAILABLE = False  # Override previous mock
    am = gpu_module.ArrayModule(backend="cupy")  # Request cupy
    assert am.backend == "cpu"  # Should fallback
    assert am.xp is np


# --- get_array_module Tests ---
def test_get_array_module_no_gpu_request():
    """"""
    assert gpu_module.get_array_module(use_gpu=False) is np


@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.GPU_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
def test_get_array_module_gpu_request_cupy():
    """"""
    sys.modules["cupy"] = MockCuPy  # Assign the class, not an instance
    gpu_module.CUPY_AVAILABLE = True  # Ensure it's set for the test
    gpu_module.GPU_AVAILABLE = True

    xp = gpu_module.get_array_module(use_gpu=True, backend="auto")
    assert xp == MockCuPy


# --- Memory Estimation and Check Tests ---
def test_estimate_gpu_memory_required():
    """"""
    shape = (1000, 1000, 3)
    attractors = 10
    mem_mb = gpu_module.estimate_gpu_memory_required(
        shape, attractors, dtype=np.float32
    )

    bytes_per_element = np.dtype(np.float32).itemsize
    expected_image_mem = shape[0] * shape[1] * shape[2] * bytes_per_element * 4
    expected_attractor_mem = attractors * shape[2] * bytes_per_element * 2
    expected_weight_mem = shape[0] * shape[1] * attractors * bytes_per_element
    expected_total_bytes = (
        expected_image_mem + expected_attractor_mem + expected_weight_mem
    ) * 1.2
    expected_mem_mb = expected_total_bytes / (1024 * 1024)

    assert np.isclose(mem_mb, expected_mem_mb)


@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
def test_check_gpu_memory_available_cupy():
    """"""
    sys.modules["cupy"] = MockCuPy()  # Ensure mock is in sys.modules
    gpu_module.CUPY_AVAILABLE = True  # Ensure it's set for the test

    # MockCuPy.cuda.Device().mem_info is (1GB free, 2GB total)
    # 1GB = 1024 MB
    has_enough, free_mb, total_mb = gpu_module.check_gpu_memory_available(
        required_mb=500
    )
    assert has_enough is True
    assert free_mb == 1000
    assert total_mb == 2000

    has_enough, _, _ = gpu_module.check_gpu_memory_available(required_mb=1500)
    assert has_enough is False


@mock.patch("imgcolorshine.gpu._check_jax_available", return_value=True)
@mock.patch("imgcolorshine.gpu.JAX_AVAILABLE", True)
def test_check_gpu_memory_available_jax(mock_check_jax):
    """"""
    # JAX path currently assumes enough memory
    gpu_module.JAX_AVAILABLE = True  # Set by mock
    sys.modules["jax"] = MockJax()  # Ensure mock is in sys.modules

    has_enough, free_mb, total_mb = gpu_module.check_gpu_memory_available(
        required_mb=500
    )
    assert has_enough is True
    assert free_mb == 0  # JAX path returns 0 for free/total
    assert total_mb == 0


def test_check_gpu_memory_no_gpu():
    """"""
    has_enough, free_mb, total_mb = gpu_module.check_gpu_memory_available(
        required_mb=500
    )
    assert has_enough is False
    assert free_mb == 0
    assert total_mb == 0


# --- GPUMemoryPool Tests ---
@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
def test_gpu_memory_pool_cupy():
    """"""
    sys.modules["cupy"] = MockCuPy()
    gpu_module.CUPY_AVAILABLE = True

    pool_instance = gpu_module.GPUMemoryPool(backend="cupy")
    assert pool_instance.backend == "cupy"
    assert isinstance(pool_instance.pool, MockCuPy.cuda.MemoryPool)

    # Mock some usage
    pool_instance.pool.used_bytes_val = 100
    pool_instance.pool.total_bytes_val = 200

    usage = pool_instance.get_usage()
    assert usage["used_bytes"] == 100
    assert usage["total_bytes"] == 200

    pool_instance.clear()  # Should call pool.free_all_blocks()


def test_gpu_memory_pool_cpu():
    """"""
    pool_instance = gpu_module.GPUMemoryPool(backend="cpu")
    assert pool_instance.backend == "cpu"
    assert pool_instance.pool is None
    assert pool_instance.get_usage() is None
    pool_instance.clear()  # Should do nothing


# Test get_memory_pool singleton behavior
@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
def test_get_memory_pool_singleton():
    """"""
    sys.modules["cupy"] = MockCuPy()
    gpu_module.CUPY_AVAILABLE = True

    gpu_module._memory_pool = None  # Reset singleton for test
    pool1 = gpu_module.get_memory_pool(backend="cupy")
    pool2 = gpu_module.get_memory_pool(backend="cupy")  # Should return same instance
    assert pool1 is pool2
    assert pool1.backend == "cupy"

    gpu_module._memory_pool = None  # Reset
    pool_cpu = gpu_module.get_memory_pool(backend="cpu")
    assert pool_cpu.backend == "cpu"


# Test _check_jax_available function
@mock.patch.dict(sys.modules, {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()})
def test_check_jax_available_success():
    """"""
    # Mock jax.devices to simulate GPU availability
    mock_jax_module = sys.modules["jax"]
    mock_jax_module.devices.return_value = ["gpu_device_1"]  # Simulate one GPU

    # Reset JAX_AVAILABLE before check
    gpu_module.JAX_AVAILABLE = False
    gpu_module.GPU_AVAILABLE = False

    assert gpu_module._check_jax_available() is True
    assert gpu_module.JAX_AVAILABLE is True
    assert gpu_module.GPU_AVAILABLE is True  # Should be set if JAX GPU is found


@mock.patch.dict(sys.modules, {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()})
def test_check_jax_available_no_gpu_device():
    """"""
    mock_jax_module = sys.modules["jax"]
    mock_jax_module.devices.return_value = []  # Simulate no GPU devices

    gpu_module.JAX_AVAILABLE = False
    assert gpu_module._check_jax_available() is False
    assert gpu_module.JAX_AVAILABLE is False


@mock.patch.dict(sys.modules, {"jax": None})  # Simulate JAX not installed
def test_check_jax_available_not_installed():
    """"""
    # Need to reload gpu_module if jax import is at module level and we want to test its absence
    # However, _check_jax_available tries to import jax inside the function
    # So, ensuring 'jax' is not in sys.modules or is None should be enough.

    if "jax" in sys.modules:
        del sys.modules["jax"]
    if "jax.numpy" in sys.modules:
        del sys.modules["jax.numpy"]

    gpu_module.JAX_AVAILABLE = False
    assert gpu_module._check_jax_available() is False
    assert gpu_module.JAX_AVAILABLE is False

    # Restore original state if necessary (though pytest fixtures should handle isolation)
    # This is more for manual script running or complex module interactions.


@mock.patch.dict(sys.modules, {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()})
def test_check_jax_numpy_compatibility_error():
    """"""
    # Simulate the specific NumPy version incompatibility error
    mock_jax_module = sys.modules["jax"]
    mock_jax_module.devices.side_effect = ImportError(
        "numpy.core._multiarray_umath failed to import"
    )

    gpu_module.JAX_AVAILABLE = False
    assert gpu_module._check_jax_available() is False
    assert gpu_module.JAX_AVAILABLE is False


# Test ArrayModule.get_info()
@mock.patch("imgcolorshine.gpu.CUPY_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.cp", MockCuPy())
def test_array_module_get_info_cupy():
    """"""
    sys.modules["cupy"] = MockCuPy()
    gpu_module.CUPY_AVAILABLE = True
    gpu_module.GPU_AVAILABLE = True

    am = gpu_module.ArrayModule(backend="cupy")
    info = am.get_info()

    assert info["backend"] == "cupy"
    assert info["gpu_available"] is True
    assert info["cuda_version"] == 11020
    assert info["device_name"] == "MockCuPyDevice"
    assert info["device_memory"] == (1024 * 1024 * 1000, 1024 * 1024 * 2000)


@mock.patch("imgcolorshine.gpu._check_jax_available", return_value=True)
@mock.patch("imgcolorshine.gpu.JAX_AVAILABLE", True)
@mock.patch("imgcolorshine.gpu.GPU_AVAILABLE", True)
def test_array_module_get_info_jax(mock_check_jax):
    """"""
    gpu_module.JAX_AVAILABLE = True
    gpu_module.GPU_AVAILABLE = True
    sys.modules["jax"] = MockJax()
    sys.modules["jax.numpy"] = MockJax.numpy

    am = gpu_module.ArrayModule(backend="jax")
    info = am.get_info()

    assert info["backend"] == "jax"
    assert info["gpu_available"] is True
    assert info["devices"] == ["MockJaxGPU"]  # As per MockJax


def test_array_module_get_info_cpu():
    """"""
    am = gpu_module.ArrayModule(backend="cpu")
    info = am.get_info()

    assert info["backend"] == "cpu"
    # GPU_AVAILABLE might be true if a lib was detected but CPU backend was forced
    # So, we check am.backend == "cpu" mainly.
    # The info["gpu_available"] reflects the global gpu_module.GPU_AVAILABLE
    assert info["gpu_available"] == gpu_module.GPU_AVAILABLE


@mock.patch.dict(sys.modules, {"cupy": mock.MagicMock()})
def test_cupy_import_generic_exception():
    """"""
    mock_cupy_module = sys.modules["cupy"]
    # Make 'import cupy as cp' succeed, but cp.cuda.is_available() raise Exception
    # Need to ensure 'cuda' attribute exists on the mock_cupy_module first
    if not hasattr(mock_cupy_module, "cuda"):
        mock_cupy_module.cuda = mock.MagicMock()
    mock_cupy_module.cuda.is_available.side_effect = Exception("Simulated CuPy error")

    # Reload gpu_module to trigger the import logic with this mock
    # Store original global state that reload might affect if not part of this test's goal
    original_jax_available = gpu_module.JAX_AVAILABLE

    importlib.reload(gpu_module)

    assert not gpu_module.CUPY_AVAILABLE
    # Check if GPU_AVAILABLE was affected only by CuPy's part of the reload
    # If JAX was previously True, it should remain so unless reload clears it before JAX check
    # For this test, we focus on CuPy's effect. If JAX was True and re-checked, it might become True again.
    # The reset_globals fixture will run after this test, cleaning up for the next.
    # Simplest is to assert that CUPY part of GPU_AVAILABLE logic is False.
    # If JAX was not available, then GPU_AVAILABLE should be False.
    if (
        not original_jax_available
    ):  # If JAX wasn't making GPU_AVAILABLE true before reload
        assert not gpu_module.GPU_AVAILABLE


@mock.patch.dict(sys.modules, {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()})
def test_jax_import_generic_exception():
    """"""
    mock_jax_module = sys.modules["jax"]
    # Make 'import jax' succeed, but jax.devices() raise Exception
    mock_jax_module.devices.side_effect = Exception("Simulated JAX error")

    # Reset JAX_AVAILABLE flags and _jax_checked before check
    gpu_module.JAX_AVAILABLE = False
    # Preserve CUPY's contribution to GPU_AVAILABLE
    original_cupy_gpu_available = gpu_module.GPU_AVAILABLE and gpu_module.CUPY_AVAILABLE
    gpu_module.GPU_AVAILABLE = original_cupy_gpu_available  # Reset based on CuPy only
    gpu_module._jax_checked = False  # Force re-check

    # Call a function that triggers _check_jax_available
    gpu_module.ArrayModule(
        backend="jax"
    )  # This will call _select_backend -> _check_jax_available

    assert not gpu_module.JAX_AVAILABLE
    # GPU_AVAILABLE should now only reflect CuPy's status
    assert original_cupy_gpu_available == gpu_module.GPU_AVAILABLE
