"""Finding, inspecting and stopping processes with the ``/proc`` file system of Linux."""

import collections.abc
import dataclasses
import os
from pathlib import Path
import signal
import time

__all__ = [
    "ProcessInfo",
    "find_descendants",
    "format_process",
    "is_alive",
    "read_cmdline",
    "read_process",
    "send_signal",
    "stop",
    "wait_for_exit",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ProcessInfo:
    """Information on a process from ``/proc/PID/stat``, see proc_pid_stat(5)."""

    pid: int
    ppid: int
    #: The state, e.g. ``R`` for running, ``S`` for sleeping and ``Z`` for zombie
    state: str
    #: The start time in clock ticks after the system boot.
    #: Together with the PID, this identifies the process even if the PID is reused.
    start_time: int


def find_descendants(roots: collections.abc.Iterable[int]) -> dict[int, ProcessInfo]:
    """Find the child processes of the given processes recursively.

    :param roots: the process IDs of the root processes
    :return: the descendant processes, excluding the root processes
    """
    children: dict[int, list[ProcessInfo]] = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and (process := read_process(int(path.name))) is not None:
            children.setdefault(process.ppid, []).append(process)
    descendants: dict[int, ProcessInfo] = {}
    stack = list(roots)
    while stack:
        for child in children.get(stack.pop(), []):
            if child.pid not in descendants:
                descendants[child.pid] = child
                stack.append(child.pid)
    return descendants


def format_process(process: ProcessInfo, max_length: int = 150) -> str:
    """Format a process as its PID and command line.

    :param process: the process
    :param max_length: the maximum length of the command line, after which it is truncated
    :return: the formatted process
    """
    cmdline = " ".join(read_cmdline(process.pid))
    if len(cmdline) > max_length:
        cmdline = cmdline[: max_length - 3] + "..."
    return f"{process.pid:>8} {cmdline}"


def is_alive(process: ProcessInfo) -> bool:
    """Check whether a process is still running.

    :param process: the process
    :return: ``False`` if the process has exited, including if it is a zombie, or if its PID has been reused
    """
    current = read_process(process.pid)
    return current is not None and current.start_time == process.start_time and current.state != "Z"


def read_cmdline(pid: int) -> list[str]:
    """Read the command-line arguments of a process.

    :param pid: the process ID
    :return: the arguments, or an empty list if the process does not exist
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [arg.decode(errors="replace") for arg in cmdline.split(b"\0") if arg]


def read_process(pid: int) -> ProcessInfo | None:
    """Read the information on a process.

    :param pid: the process ID
    :return: the information, or ``None`` if the process does not exist
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name is in parentheses and may contain spaces and parentheses itself,
    # so the fields are counted from the last closing parenthesis.
    # The fields after it are numbered from 3 in proc_pid_stat(5).
    fields = stat[stat.rindex(")") + 2 :].split()
    return ProcessInfo(pid=pid, ppid=int(fields[1]), state=fields[0], start_time=int(fields[19]))


def send_signal(processes: collections.abc.Iterable[ProcessInfo], sig: signal.Signals) -> int:
    """Send a signal to those of the processes that are still running.

    :param processes: the processes
    :param sig: the signal
    :return: the number of processes to which the signal was sent
    """
    count = 0
    for process in processes:
        if is_alive(process):
            try:
                os.kill(process.pid, sig)
            except ProcessLookupError:
                continue
            count += 1
    return count


def stop(targets: dict[int, ProcessInfo], timeout: float) -> list[ProcessInfo]:
    """Stop processes, and their child processes that are created while waiting for them to exit.

    First, SIGTERM is sent to the processes in the given order,
    and those that are still running after the timeout are killed with SIGKILL.
    The processes are identified by their PIDs and start times,
    so that the processes that have been reparented to init after the exit of their parents are also stopped,
    and that no unrelated processes are signalled, even if their PIDs have been reused.
    The progress is printed to stdout.

    :param targets: the processes by their PIDs, e.g. a process and its descendants from :func:`find_descendants`.
        The child processes that are created while waiting are added to these.
    :param timeout: the time to wait after SIGTERM before sending SIGKILL, in seconds
    :return: the processes that could not be stopped
    """
    sent = send_signal(targets.values(), signal.SIGTERM)
    print(f"Sent SIGTERM to {sent} processes. Waiting up to {timeout} s for them to exit.")
    alive = wait_for_exit(targets, timeout, signal.SIGTERM)
    if alive:
        print(f"Sending SIGKILL to {len(alive)} processes that did not exit:")
        for target in alive:
            print(format_process(target))
        send_signal(alive, signal.SIGKILL)
        alive = wait_for_exit(targets, 5, signal.SIGKILL)
    if alive:
        print(f"Failed to stop {len(alive)} processes:")
        for target in alive:
            print(format_process(target))
    return alive


def wait_for_exit(targets: dict[int, ProcessInfo], timeout: float, sig: signal.Signals) -> list[ProcessInfo]:
    """Wait for the processes to exit.

    The child processes that are created in the meantime are added to the targets, and the signal is sent to them.

    :param targets: the processes, which are updated in place
    :param timeout: the maximum waiting time in seconds
    :param sig: the signal to send to the new child processes
    :return: the processes that are still running
    """
    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in targets.values() if is_alive(process)]
        new = {
            pid: process
            for pid, process in find_descendants(process.pid for process in alive).items()
            if pid not in targets
        }
        targets.update(new)
        send_signal(new.values(), sig)
        alive += new.values()
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.1)
