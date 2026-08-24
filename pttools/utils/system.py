"""System utilities"""

import multiprocessing
import os
import platform
import subprocess
import sys
import sysconfig

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

#: The number of available CPU cores
AVAILABLE_CPU_CORES: int | None
#: Whether the system provides information on which CPU cores are available for this process
CPU_AFFINITY: bool = False
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
#: Whether running on the Read the Docs builder
IS_READ_THE_DOCS: bool = "READTHEDOCS_VIRTUALENV_PATH" in os.environ
#: PTtools installation directory
PTTOOLS_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The method used by `multiprocessing` to start parallel processes
PROCESS_START_METHOD: str = multiprocessing.get_start_method()
#: Whether this Python installation supports free threading
SUPPORTS_FREETHREADING: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
#: Whether this Python installation supports :py:class:`multiprocessing.InterpreterPoolExecutor`
SUPPORTS_INTERPRETER_POOL: bool = hasattr(multiprocessing, "InterpreterPoolExecutor")
#: Uname information of the current platform
UNAME: platform.uname_result = platform.uname()

# Constants determined by other constants
#: Whether parallel processes are started by forking
FORKING: bool = PROCESS_START_METHOD == "fork"
#: Whether PTtools is installed as a pip package
IS_PIP_PACKAGE: bool = os.path.basename(os.path.dirname(PTTOOLS_DIR)) == "site-packages"

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
    """Get the last n lines from dmesg

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
            stderr=subprocess.STDOUT
        )
        output = process.stdout.decode("utf-8")
    except Exception as err:
        return f"Running dmesg failed: {err}"
    if print_output:
        print(f"Last {n_lines} lines from dmesg:")
        print(output)
    return output


def platform_info() -> str:
    return (
        f"OS: {UNAME.system} ({UNAME.release}), CPU: {UNAME.processor} ({UNAME.machine}), "
        f"Python: {platform.python_version()}, "
        f"Start method: {PROCESS_START_METHOD}, available: {multiprocessing.get_all_start_methods()}."
    )


def psutil_info() -> str:
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
    ) if ram.percent > 80 else ""
    msg = (
        f"{msg_cpu} RAM use: {ram.used * 1e-9:.2f} / {ram.total * 1e-9:.2f} GB = {ram.percent} %, "
        f"available {ram.available} GB.{msg_ram_high}"
    )
    return msg


def system_info() -> str:
    try:
        dmesg_msg = rf"Dmesg output:\n{dmesg()}"
    except Exception as err:
        dmesg_msg = str(err)

    return f"{platform_info()} {psutil_info()} {dmesg_msg}"
