"""Performance profiling script

When implementing new tests, the functions should be called at least once to JIT-compile them before profiling
"""

import cProfile
import io
import os
import pstats
import unittest

import numpy as np

import pttools.ssmtools as ssm
from tests.paper.test_pow_specs import TestPowSpecs

PROFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test-results", "profiles")
if not os.path.isdir(PROFILE_DIR):
    os.mkdir(PROFILE_DIR)


def process_stats(name: str, pr: cProfile.Profile):
    path = os.path.join(PROFILE_DIR, f"{name}")
    pr.dump_stats(f"{path}.pstat")

    # Save to file
    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("tottime")
    ps.print_stats()
    text = stream.getvalue()
    print(text)

    with open(f"{path}.txt", "w") as file:
        file.write(text)


class TestProfile(unittest.TestCase):
    @staticmethod
    def test_profile_gw():
        z = np.logspace(0, 2, 100)
        ssm.power_gw_scaled(z, [0.1, 0.1])

        pr = cProfile.Profile()
        pr.enable()
        ssm.power_gw_scaled(z, [0.1, 0.1])
        pr.disable()
        process_stats("gw", pr)

    @staticmethod
    def test_profile_pow_specs():
        pr = cProfile.Profile()
        pr.enable()
        TestPowSpecs.test_pow_specs()
        pr.disable()
        process_stats("pow_specs", pr)


if __name__ == "__main__":
    unittest.main()
