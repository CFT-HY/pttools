"""System utilities"""

import multiprocessing
import os
import platform
import subprocess
import sys

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

AVAILABLE_CPU_CORES: int | None
START_METHOD: str = multiprocessing.get_start_method()
FORKING: bool = START_METHOD == "fork"
UNAME: platform.uname_result = platform.uname()
CPU_AFFINITY: bool = False
IS_GITHUB_ACTIONS: bool = "GITHUB_ACTIONS" in os.environ
IS_LINUX: bool = sys.platform.startswith('linux')
IS_OSX: bool = sys.platform.startswith('darwin')
IS_WINDOWS: bool = sys.platform.startswith('win32')
IS_READ_THE_DOCS: bool = "READTHEDOCS_VIRTUALENV_PATH" in os.environ

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
        f"Start method: {START_METHOD}, available: {multiprocessing.get_all_start_methods()}."
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
