# -*- coding: utf-8 -*-
import unittest
from vectornaut.synthesizer import Synthesizer

class TestSynthesisUnit(unittest.TestCase):
    def test_mock_synthesis(self):
        synth = Synthesizer()
        report = synth.mock_generate_synthesis()
        
        self.assertIsNotNone(report.mechanical_limits)
        self.assertIsNotNone(report.manufacturing_methods)
        self.assertIsNotNone(report.cost_estimation)
        self.assertIsNotNone(report.validation_experiments)
        self.assertIsNotNone(report.industry_partners)
        
        self.assertTrue("MPa" in report.mechanical_limits)
        self.assertTrue("Partner" in report.industry_partners)

if __name__ == "__main__":
    unittest.main()
