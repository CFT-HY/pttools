"""Unit tests for the watchdogs of the documentation build."""

import subprocess
import sys
import textwrap
import unittest


def run_script(script: str) -> subprocess.CompletedProcess[str]:
    """Run the script in a new Python process, since the watchdog stops the process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=60, check=False
    )


class WatchdogTest(unittest.TestCase):
    """Tests for :py:func:`pttools.docs.setup.start_watchdog` and :py:func:`pttools.docs.setup.example_watchdog`."""

    def test_start_watchdog(self) -> None:
        """The watchdog stops the process and prints the stack trace."""
        result = run_script("""
            import time
            from pttools.docs.setup import start_watchdog
            start_watchdog(0.5)
            time.sleep(30)
            print("not stopped")
        """)
        assert result.returncode == 1
        assert "not stopped" not in result.stdout
        assert "Timeout" in result.stderr
        assert "<module>" in result.stderr

    def test_example_watchdog_stops(self) -> None:
        """The watchdog of an example stops the process."""
        result = run_script("""
            import time
            from pttools.docs import setup
            from pttools.docs.setup import example_watchdog
            setup.EXAMPLE_TIMEOUT = 0.5
            example_watchdog({}, None, "before")
            time.sleep(30)
            print("not stopped")
        """)
        assert result.returncode == 1
        assert "not stopped" not in result.stdout
        assert "Timeout" in result.stderr

    def test_example_watchdog_cancel(self) -> None:
        """The watchdog of an example is cancelled after the example."""
        result = run_script("""
            import time
            from pttools.docs import setup
            from pttools.docs.setup import example_watchdog
            setup.EXAMPLE_TIMEOUT = 0.5
            example_watchdog({}, None, "before")
            example_watchdog({}, None, "after")
            time.sleep(1)
            print("finished")
        """)
        assert result.returncode == 0, result.stderr
        assert "finished" in result.stdout

    def test_example_watchdog_restores_build_watchdog(self) -> None:
        """The watchdog of the build is restored after an example that is run in the same process."""
        result = run_script("""
            import time
            from pttools.docs import setup
            from pttools.docs.setup import example_watchdog, start_watchdog
            setup.EXAMPLE_TIMEOUT = 100
            start_watchdog(1)
            example_watchdog({}, None, "before")
            example_watchdog({}, None, "after")
            time.sleep(30)
            print("not stopped")
        """)
        assert result.returncode == 1
        assert "not stopped" not in result.stdout
        assert "Timeout" in result.stderr

    def test_example_watchdog_disabled(self) -> None:
        """The watchdog of an example is disabled outside Read the Docs."""
        result = run_script("""
            import os
            import time
            os.environ.pop("READTHEDOCS_VIRTUALENV_PATH", None)
            from pttools.docs import setup
            from pttools.docs.setup import example_watchdog
            assert setup.EXAMPLE_TIMEOUT is None
            example_watchdog({}, None, "before")
            time.sleep(1)
            print("finished")
        """)
        assert result.returncode == 0, result.stderr
        assert "finished" in result.stdout


if __name__ == "__main__":
    unittest.main()
