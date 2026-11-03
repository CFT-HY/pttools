"""System utilities."""

import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from pttools.logging import log_console
from pttools.type_hints import IS_DOCS

# IS_DOCS is imported on purpose, so that it's available with the other system constants.
__all__ = [
    "AVAILABLE_CPU_CORES",
    "CPU_AFFINITY",
    "CPU_ARCH",
    "FORKING",
    "IS_ARM64",
    "IS_CFT_BIG_MACHINE",
    "IS_DOCS",
    "IS_GITHUB_ACTIONS",
    "IS_LINUX",
    "IS_OSX",
    "IS_PACKAGE",
    "IS_READ_THE_DOCS",
    "IS_WINDOWS",
    "IS_X86_64",
    "PROCESS_START_METHOD",
    "PROJECT_DIR",
    "PTTOOLS_DIR",
    "RAM_USE_HIGH_PERCENT",
    "SUPPORTS_FREETHREADING",
    "SUPPORTS_INTERPRETER_POOL",
    "UNAME",
    "dmesg",
    "physical_cpu_count",
    "platform_info",
    "project_dir",
    "psutil_info",
    "system_info",
    "testing",
    "testing_or_ci",
]

#: Whether currently running tests with pytest. Use :py:func:`pttools.utils.system.testing` to get this value.
_TESTING: bool = False
#: The number of available CPU cores
AVAILABLE_CPU_CORES: int | None
#: Whether the system provides information on which CPU cores are available for this process
CPU_AFFINITY: bool = False
#: CPU architecture as reported by :func:`platform.machine`, e.g. "x86_64" or "aarch64".
#: The names depend on the operating system, so use :data:`IS_ARM64` and :data:`IS_X86_64` to check for them.
CPU_ARCH: str = platform.machine()
#: Whether the CPU architecture is 64-bit ARM.
#: Python reports it as "aarch64" on Linux, as "arm64" on macOS, and as "ARM64" on Windows.
IS_ARM64: bool = CPU_ARCH in ("aarch64", "arm64", "ARM64")
#: Whether running on the CFT Big Machine
IS_CFT_BIG_MACHINE: bool = platform.node() == "dx2-528-26839.ad.helsinki.fi"
#: Whether running on GitHub Actions
IS_GITHUB_ACTIONS: bool = "GITHUB_ACTIONS" in os.environ
#: Whether running on Linux
IS_LINUX: bool = sys.platform.startswith('linux')
#: Whether running on macOS
IS_OSX: bool = sys.platform.startswith('darwin')
#: Whether running on Windows
IS_WINDOWS: bool = sys.platform.startswith('win32')
#: Whether the CPU architecture is x86-64.
#: Python reports it as "x86_64" on Linux and macOS, and as "AMD64" on Windows.
IS_X86_64: bool = CPU_ARCH in ("x86_64", "AMD64")
#: Whether running on the Read the Docs builder
IS_READ_THE_DOCS: bool = "READTHEDOCS_VIRTUALENV_PATH" in os.environ
#: PTtools installation directory
PTTOOLS_DIR: Path = Path(__file__).resolve().parent.parent
#: The method used by `multiprocessing` to start parallel processes
PROCESS_START_METHOD: str = multiprocessing.get_start_method()
#: RAM use percentage above which a warning is included in :py:func:`psutil_info`
RAM_USE_HIGH_PERCENT: float = 80
#: Whether this Python installation supports free threading
SUPPORTS_FREETHREADING: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
#: Whether this Python installation supports :py:class:`concurrent.futures.InterpreterPoolExecutor`,
#: which is available in Python 3.14 and later
SUPPORTS_INTERPRETER_POOL: bool = hasattr(concurrent.futures, "InterpreterPoolExecutor")
#: Uname information of the current platform
UNAME: platform.uname_result = platform.uname()

# Constants determined by other constants
#: Whether parallel processes are started by forking
FORKING: bool = PROCESS_START_METHOD == "fork"
#: Whether PTtools is installed as a Python package
IS_PACKAGE: bool = PTTOOLS_DIR.parent.name == "site-packages"


def project_dir(pttools_dir: Path = PTTOOLS_DIR, is_package: bool = IS_PACKAGE) -> Path | None:
    """Root directory of the project repository.

    If PTtools is installed as a package, it's assumed to be installed in a virtual environment (e.g. ``.venv``)
    at the root of the repository of the project that uses it, such as PTPlot.
    The virtual environment is recognised by its ``pyvenv.cfg`` file,
    which is created by ``venv``, ``virtualenv`` and ``uv``.
    Otherwise, PTtools is run from its source repository, which is returned.

    :param pttools_dir: the installation directory of PTtools, see :py:data:`PTTOOLS_DIR`
    :param is_package: whether PTtools is installed as a package, see :py:data:`IS_PACKAGE`
    :return: the root directory of the project repository,
        or None if PTtools is installed as a package, but not in a virtual environment
    """
    if not is_package:
        return pttools_dir.parent
    for parent in pttools_dir.parents:
        if (parent / "pyvenv.cfg").is_file():
            return parent.parent
    return None


#: Root directory of the project repository, see :py:func:`project_dir`
PROJECT_DIR: Path | None = project_dir()

# This is the same implementation as in Numba.
# https://numba.readthedocs.io/en/stable/user/threading-layer.html
try:
    # This is available only on some platforms
    AVAILABLE_CPU_CORES = len(os.sched_getaffinity(0))
    CPU_AFFINITY = True
except AttributeError:
    # multiprocessing.cpu_count() is a wrapper around os.cpu_count()
    # https://stackoverflow.com/a/53537394
    AVAILABLE_CPU_CORES = os.cpu_count()


def dmesg(n_lines: int = 100, print_output: bool = False) -> str:
    """Get the last n lines from dmesg.

    :param n_lines: Number of lines to retrieve from dmesg
    :param print_output: Whether to print the output to the console
    :return: The last n lines from dmesg as a string
    """
    if not IS_LINUX:
        raise OSError("Cannot run dmesg since not running on a Linux system.")
    if os.geteuid() != 0:
        raise PermissionError("Cannot run dmesg since not running as root.")
    try:
        process = subprocess.run(
            ["dmesg", "|", "tail", "-n", str(n_lines)],
            capture_output=True,
            check=False,
            stderr=subprocess.STDOUT
        )
        output = process.stdout.decode("utf-8")
    except Exception as err:  # noqa: BLE001
        return f"Running dmesg failed: {err}"
    if print_output:
        print(f"Last {n_lines} lines from dmesg:")
        print(output)
    return output


def physical_cpu_count() -> int:
    """Get the number of physical CPU cores available to this process.

    On Linux, the logical CPUs to which the process is restricted are found with :func:`os.sched_getaffinity`,
    and the physical cores to which they belong from ``/sys/devices/system/cpu/cpu*/topology/core_cpus_list``,
    which lists the logical CPUs that share a physical core with simultaneous multithreading (SMT).
    If this is not available, the physical cores of the whole system are counted with :func:`psutil.cpu_count`,
    if :mod:`psutil` is installed.
    If neither is available, an error is logged, and the number of logical CPUs available to the process is returned.

    :return: the number of physical CPU cores
    """
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    try:
        return len({
            Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list").read_text().strip() for cpu in cpus
        })
    except OSError:
        pass
    if psutil is None:
        log_console(
            f"ERROR: The CPU topology is not available in /sys, and psutil is not installed for {sys.executable}, "
            f"so the number of physical CPU cores is unknown. Using the number of logical CPUs ({len(cpus)}) instead."
        )
        return len(cpus)
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        log_console(
            "ERROR: The CPU topology is not available in /sys, and psutil could not determine "
            f"the number of physical CPU cores. Using the number of logical CPUs ({len(cpus)}) instead."
        )
        return len(cpus)
    return cores


def platform_info() -> str:
    """Get information about the operating system, the CPU, the Python version and the process start methods."""
    return (
        f"OS: {UNAME.system} ({UNAME.release}), CPU: {UNAME.processor} ({UNAME.machine}), "
        f"Python: {platform.python_version()}, "
        f"Start method: {PROCESS_START_METHOD}, available: {multiprocessing.get_all_start_methods()}."
    )


def psutil_info() -> str:
    """Get information about the CPU and RAM use with psutil, including a warning if the RAM use is high."""
    if psutil is None:
        return "Please install psutil for more info."
    cpu = psutil.getloadavg()
    cpu_count = psutil.cpu_count()
    msg_cpu = "Could not determine the number of CPU cores." if cpu_count is None else (
        f"CPU cores: {cpu_count}, CPU use: "
        f"1 min {cpu[0] / cpu_count * 100} %, "
        f"5 min {cpu[1] / cpu_count * 100} %, "
        f"15 min {cpu[2] / cpu_count * 100} %."
    )
    ram = psutil.virtual_memory()
    msg_ram_high = (
        " RAM use is high. "
        "Please reduce the number of worker processes or close applications running in the background."
    ) if ram.percent > RAM_USE_HIGH_PERCENT else ""
    return (
        f"{msg_cpu} RAM use: {ram.used * 1e-9:.2f} / {ram.total * 1e-9:.2f} GB = {ram.percent} %, "
        f"available {ram.available * 1e-9:.2f} GB.{msg_ram_high}"
    )


def system_info() -> str:
    """Get system information for debugging, including the output of dmesg if it's available."""
    try:
        dmesg_msg = rf"Dmesg output:\n{dmesg()}"
    except Exception as err:  # noqa: BLE001
        dmesg_msg = str(err)

    return f"{platform_info()} {psutil_info()} {dmesg_msg}"


def testing() -> bool:
    """Get whether currently running unit tests."""
    return _TESTING


def testing_or_ci() -> bool:
    """Get whether currently running unit tests or a CI workflow."""
    return testing() or IS_GITHUB_ACTIONS
