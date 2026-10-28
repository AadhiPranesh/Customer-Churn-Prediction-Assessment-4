"""
Automated unit test running the comprehensive quality gate audit.
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_verification_check import run_checks

class TestQualityGate(unittest.TestCase):
    def test_full_quality_gate(self):
        self.assertTrue(run_checks())

if __name__ == '__main__':
    unittest.main()
