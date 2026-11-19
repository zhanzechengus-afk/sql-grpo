import os
import shutil
import subprocess
import unittest

from src.grpo_rewards import code_test_outcomes


class TestDockerCodeReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = os.environ.get("SQL_GRPO_TEST_IMAGE", "python:3.11-slim")
        if not shutil.which("docker"):
            raise unittest.SkipTest("Docker CLI unavailable")
        try:
            subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=15)
            subprocess.run(["docker", "image", "inspect", cls.image], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=15)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise unittest.SkipTest(f"local Docker daemon or image {cls.image} unavailable") from exc

    def test_assert_function_and_stdin(self):
        cases = [
            ("def double(x): return 2*x", [{"kind":"assert", "code":"assert double(2) == 4"},
                                          {"kind":"assert", "code":"assert double(2) == 8"}]),
            ("class Solution:\n def double(self,x): return 2*x",
             [{"kind":"function","function_name":"double","input":[3],"expected":6},
              {"kind":"function","function_name":"double","input":[3],"expected":4}]),
            ("print(int(input()) * 2)", [{"kind":"stdin","input":"3\n","expected":"6\n"},
                                          {"kind":"stdin","input":"4\n","expected":"9\n"}]),
        ]
        for program, tests in cases:
            with self.subTest(kind=tests[0]["kind"]):
                results, parsed = code_test_outcomes(program, {"tests":tests}, image=self.image)
                self.assertTrue(parsed)
                self.assertEqual(results, [True, False])
