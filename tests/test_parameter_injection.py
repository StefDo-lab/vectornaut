# -*- coding: utf-8 -*-
"""Offline regression tests for auto-injected solver parameters.

The solver dispatcher injects default values for symbols that appear in the
governing equation / boundary conditions but are missing from the audited
parameters. Those injected values must actually reach the 1D solver and must
be recorded on the AuditorOutput (history, synthesis and the API response read
``audited_parameters_dict``).
"""
import unittest

from vectornaut.config import (
    AuditedParameter,
    AuditorOutput,
    DimensionlessNumber,
    MinerOutput,
    ParameterProposal,
)
from vectornaut.solver_dispatcher import dispatch_and_solve


UI_METADATA = {
    "domain_name": "Fluid Dynamics",
    "independent_var": {"label": "Channel Height", "unit": "m"},
    "dependent_var": {"label": "Flow Velocity", "unit": "m/s"},
    "primary_metric": {"label": "Wall Shear Stress"},
    "reference_metric": {"label": "Analytical Wall Shear Stress"},
    "performance_gain": {"label": "Drag Reduction Efficiency"},
}


def _miner_1d(governing_equation, boundary_conditions):
    return MinerOutput(
        design_name="Injection Test Channel",
        inspiration_source="Testorganismus (Prüfling)",
        domain="Fluid Dynamics",
        physical_mechanism="Druckgetriebene Strömung zwischen zwei Platten.",
        parameters=[
            ParameterProposal(name="viscosity", value=0.001, min_bound=0.0001, max_bound=0.01, justification="Water"),
        ],
        governing_equation=governing_equation,
        boundary_conditions=boundary_conditions,
        independent_variables=["y"],
        dependent_variables=["u"],
        svg_schematic="<svg></svg>",
    )


def _auditor(solver_method, audited=None, ui_metadata=None):
    return AuditorOutput(
        audit_passed=True,
        audit_notes="Audit bestanden.",
        audited_parameters=[
            AuditedParameter(name=name, value=value)
            for name, value in (audited or {"viscosity": 0.001}).items()
        ],
        dimensionless_numbers=[DimensionlessNumber(name="ReynoldsNumber", value=100.0)],
        simulation_coefficient=0.0,
        solver_method=solver_method,
        ui_metadata=ui_metadata or UI_METADATA,
    )


class TestParameterInjection1D(unittest.TestCase):
    def test_injected_parameter_reaches_analytical_solver(self):
        # 'forcing_term' is referenced by the equation but was never audited.
        miner = _miner_1d("d2u_dy2 = -forcing_term", ["u(0) = 0", "u(1) = 0"])
        auditor = _auditor("analytical")

        sim = dispatch_and_solve(miner, auditor, epochs=10)

        self.assertEqual(sim.solver_method, "analytical")
        # u'' = -1 with u(0) = u(1) = 0  ->  u = y(1 - y)/2, u'(0) = 0.5
        self.assertAlmostEqual(sim.primary_metric_value, 0.5, places=6)
        self.assertAlmostEqual(max(sim.solution_primary), 0.125, delta=2e-3)
        self.assertLess(sim.relative_error, 1e-6)

    def test_injected_parameter_reaches_scipy_solver(self):
        miner = _miner_1d("d2u_dy2 = -forcing_term", ["u(0) = 0", "u(1) = 0"])
        auditor = _auditor("scipy")

        sim = dispatch_and_solve(miner, auditor, epochs=10)

        self.assertEqual(sim.solver_method, "scipy")
        self.assertAlmostEqual(sim.primary_metric_value, 0.5, places=3)

    def test_injected_parameter_is_recorded_on_auditor_output(self):
        miner = _miner_1d("d2u_dy2 = -forcing_term", ["u(0) = 0", "u(1) = 0"])
        auditor = _auditor("analytical")

        dispatch_and_solve(miner, auditor, epochs=10)

        audited = auditor.audited_parameters_dict
        self.assertEqual(audited.get("forcing_term"), 1.0)
        self.assertEqual(audited.get("viscosity"), 0.001)
        # Internal coefficient aliases are solver inputs, not audited parameters.
        for alias in ("simulation_coefficient", "slippage_coefficient", "lambda", "slip_length"):
            self.assertNotIn(alias, audited)
        # The injection is visible in the serialized output used for history.
        dumped_names = [p["name"] for p in auditor.model_dump()["audited_parameters"]]
        self.assertIn("forcing_term", dumped_names)
        self.assertIn("forcing_term", auditor.audit_notes)

    def test_repeated_solves_do_not_duplicate_injected_parameters(self):
        miner = _miner_1d("d2u_dy2 = -forcing_term", ["u(0) = 0", "u(1) = 0"])
        auditor = _auditor("analytical")

        dispatch_and_solve(miner, auditor, epochs=10)
        auditor.solver_method = "scipy"
        dispatch_and_solve(miner, auditor, epochs=10)

        names = [p.name for p in auditor.audited_parameters]
        self.assertEqual(names.count("forcing_term"), 1)
        self.assertEqual(auditor.audit_notes.count("forcing_term"), 1)

    def test_no_injection_leaves_auditor_output_untouched(self):
        miner = _miner_1d("d2u_dy2 = -viscosity * 1000", ["u(0) = 0", "u(1) = 0"])
        auditor = _auditor("analytical")
        before = auditor.model_dump()

        dispatch_and_solve(miner, auditor, epochs=10)

        self.assertEqual(auditor.model_dump(), before)

    def test_audited_parameters_dict_stays_a_read_only_view(self):
        auditor = _auditor("analytical")
        view = auditor.audited_parameters_dict
        view["not_audited"] = 42.0
        self.assertNotIn("not_audited", auditor.audited_parameters_dict)

    def test_add_injected_parameters_keeps_existing_values(self):
        auditor = _auditor("analytical", audited={"viscosity": 0.001})
        added = auditor.add_injected_parameters({"viscosity": 5.0, "forcing_term": 2.0})
        self.assertEqual(added, ["forcing_term"])
        self.assertEqual(auditor.audited_parameters_dict, {"viscosity": 0.001, "forcing_term": 2.0})
        self.assertEqual(auditor.add_injected_parameters({}), [])


class TestParameterInjection2D(unittest.TestCase):
    def test_injected_parameter_is_recorded_for_2d_fdm(self):
        miner = MinerOutput(
            design_name="Injection Test Plate",
            inspiration_source="Testorganismus",
            domain="Thermodynamics",
            physical_mechanism="Wärmeleitung mit Quellterm.",
            parameters=[
                ParameterProposal(name="T_hot", value=300.0, min_bound=273.0, max_bound=400.0, justification="Hot"),
                ParameterProposal(name="T_cold", value=260.0, min_bound=200.0, max_bound=300.0, justification="Cold"),
            ],
            governing_equation="d2T_dx2 + d2T_dy2 = -volumetric_source",
            boundary_conditions=[
                "T(0, y) = T_hot",
                "T(1, y) = T_cold",
                "dT_dy(x, 0) = 0",
                "dT_dy(x, 1) = 0",
            ],
            independent_variables=["x", "y"],
            dependent_variables=["T"],
            svg_schematic="<svg></svg>",
        )
        auditor = _auditor(
            "fdm",
            audited={"T_hot": 300.0, "T_cold": 260.0},
            ui_metadata={
                "domain_name": "Thermodynamics",
                "independent_var": {"label": "Position (x, y)", "unit": "m"},
                "dependent_var": {"label": "Temperature (T)", "unit": "K"},
                "primary_metric": {"label": "FDM Mean Temperature"},
                "reference_metric": {"label": "FDM Mean Temperature"},
                "performance_gain": {"label": "Insulation Efficiency"},
            },
        )

        sim = dispatch_and_solve(miner, auditor, epochs=10)

        self.assertEqual(sim.solver_method, "fdm")
        self.assertEqual(len(sim.solution_primary), 400)
        self.assertEqual(auditor.audited_parameters_dict.get("volumetric_source"), 1.0)
        self.assertEqual(auditor.audited_parameters_dict.get("T_hot"), 300.0)


if __name__ == "__main__":
    unittest.main()
