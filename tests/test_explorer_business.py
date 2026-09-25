# -*- coding: utf-8 -*-
"""Business profile: unit economics, sanity flags, critic-adjusted scoring, candidate checks."""
import math
import os
import tempfile
import unittest
from unittest import mock

from vectornaut.explorer.archive import Archive
from vectornaut.explorer.generator import INVALID, OK, REJECTED, TARGET_INFEASIBLE, check_batch
from vectornaut.explorer.profiles.base import EvaluationContext, PreparedCandidate
from vectornaut.explorer.profiles.business import (
    BUSINESS_SPACE,
    BusinessProfile,
    critic_penalty,
    sanity_flags,
    score_business,
    unit_economics,
)
from vectornaut.explorer.schemas import (
    BackOfEnvelope,
    BusinessCandidate,
    BusinessCandidateBatch,
    CriticBatch,
    CriticReview,
    DescriptorAssignment,
    UnitEconomicsInputs,
)

DESCRIPTORS = {
    "customer_segment": "smb", "revenue_model": "subscription", "market_scale": "national",
    "advantage_type": "cost", "capital_intensity": "seed",
}
# Hand-checked example (see the expected values in the tests).
BASE_INPUTS = dict(
    monthly_revenue_per_customer=100.0, gross_margin=0.8, cac=400.0, monthly_churn=0.05,
    addressable_customers=100000.0, reachable_share_3y=0.05, fixed_costs_per_year=200000.0, upfront_capex=100000.0,
)


def inputs(**changes):
    return UnitEconomicsInputs(**{**BASE_INPUTS, **changes})


def candidate(order_id="r001-01", title="Shared cold room", descriptors=None, **changes):
    fields = dict(
        order_id=order_id, title=title, summary="s",
        descriptors=[DescriptorAssignment(axis=k, value=v) for k, v in (descriptors or DESCRIPTORS).items()],
        back_of_envelope=BackOfEnvelope(quantity="revenue", formula="5000 x 100 x 12", value=6e6, unit="EUR"),
        main_risk="restaurants do not switch", novelty_vs_known="n", value_proposition="v",
        target_customer="restaurants", advantage_mechanism="pooling", inputs=inputs(), input_assumptions="a",
    )
    fields.update(changes)
    return BusinessCandidate(**fields)


class UnitEconomicsTest(unittest.TestCase):
    def test_formulas_match_hand_computed_numbers(self):
        econ = unit_economics(BASE_INPUTS)
        # gp = 100 * 0.8 = 80 EUR/month; lifetime = 1/0.05 = 20 months; LTV = 1600 EUR.
        self.assertAlmostEqual(econ["gross_profit_per_customer_month"], 80.0)
        self.assertAlmostEqual(econ["lifetime_months"], 20.0)
        self.assertAlmostEqual(econ["ltv"], 1600.0)
        self.assertAlmostEqual(econ["ltv_cac"], 4.0)
        self.assertAlmostEqual(econ["payback_months"], 5.0)
        # N = 100000 * 0.05 = 5000; customer-months = 18 * 5000 = 90000; revenue = 9.0 MEUR.
        self.assertAlmostEqual(econ["customers_3y"], 5000.0)
        self.assertAlmostEqual(econ["revenue_3y"], 9_000_000.0)
        self.assertAlmostEqual(econ["gross_profit_3y"], 7_200_000.0)
        # acquisitions = 5000 * (1 + 18 * 0.05) = 9500 -> 3.8 MEUR acquisition spend.
        self.assertAlmostEqual(econ["acquisitions_3y"], 9500.0)
        self.assertAlmostEqual(econ["acquisition_spend_3y"], 3_800_000.0)
        # profit = 7.2 - 3.8 - 3 * 0.2 - 0.1 = 2.7 MEUR; ROI = 2.7 / 4.5 = 0.6.
        self.assertAlmostEqual(econ["profit_3y"], 2_700_000.0)
        self.assertAlmostEqual(econ["roi_3y"], 0.6)
        # contribution per active customer = 12 * 80 - 12 * 0.05 * 400 = 720; break-even = 200000 / 720.
        self.assertAlmostEqual(econ["annual_contribution_per_customer"], 720.0)
        self.assertAlmostEqual(econ["break_even_customers"], 200000.0 / 720.0)

    def test_score_matches_hand_computed_components(self):
        result = score_business(inputs(), DESCRIPTORS)
        c = result.breakdown["components"]
        self.assertAlmostEqual(c["ltv_cac"], 0.75)                  # (4 - 1) / 4
        self.assertAlmostEqual(c["payback"], 1 - 5 / 36, places=6)  # 0.861111
        self.assertAlmostEqual(c["roi"], 0.8)                       # (0.6 + 1) / 2
        self.assertAlmostEqual(c["scale"], (math.log10(9e6) - 5) / 4, places=6)
        expected = 100 * (0.3 * 0.75 + 0.2 * (1 - 5 / 36) + 0.3 * 0.8 + 0.2 * (math.log10(9e6) - 5) / 4)
        self.assertAlmostEqual(result.score, expected, places=3)
        self.assertAlmostEqual(result.score, 73.4934, places=3)
        self.assertEqual(result.breakdown["sanity_flags"], [])
        self.assertEqual(result.breakdown["inputs_source"], "generator")

    def test_lifetime_is_capped_and_zero_churn_is_flagged(self):
        econ = unit_economics({**BASE_INPUTS, "monthly_churn": 0.0})
        self.assertAlmostEqual(econ["lifetime_months"], 60.0)
        codes = [f["code"] for f in sanity_flags({**BASE_INPUTS, "monthly_churn": 0.0}, DESCRIPTORS)]
        self.assertIn("churn_non_positive", codes)
        # LTV/CAC = 80 * 60 (capped lifetime) / 100 = 48 > 20
        self.assertIn("ltv_cac_implausible",
                      [f["code"] for f in sanity_flags({**BASE_INPUTS, "cac": 100.0, "monthly_churn": 0.001}, DESCRIPTORS)])

    def test_no_profit_means_infinite_break_even(self):
        econ = unit_economics({**BASE_INPUTS, "cac": 10000.0})
        self.assertLess(econ["annual_contribution_per_customer"], 0)
        self.assertEqual(econ["break_even_customers"], math.inf)


class SanityFlagTest(unittest.TestCase):
    def codes(self, descriptors=DESCRIPTORS, **changes):
        return [f["code"] for f in sanity_flags({**BASE_INPUTS, **changes}, descriptors)]

    def test_plausible_inputs_have_no_flags(self):
        self.assertEqual(self.codes(), [])

    def test_each_implausible_input_is_flagged(self):
        self.assertIn("churn_ge_100pct", self.codes(monthly_churn=1.0))
        self.assertIn("cac_non_positive", self.codes(cac=0.0))
        self.assertIn("reachable_share_gt_30pct", self.codes(reachable_share_3y=0.5))
        self.assertIn("margin_out_of_range", self.codes(gross_margin=1.2))
        hardware = {**DESCRIPTORS, "revenue_model": "hardware_plus_service"}
        self.assertIn("hardware_margin_gt_95pct", self.codes(descriptors=hardware, gross_margin=0.96))
        self.assertNotIn("hardware_margin_gt_95pct", self.codes(gross_margin=0.96))
        self.assertIn("scale_mismatch", self.codes(descriptors={**DESCRIPTORS, "market_scale": "global"},
                                                   addressable_customers=500.0))
        self.assertIn("capital_mismatch", self.codes(descriptors={**DESCRIPTORS, "capital_intensity": "bootstrap"},
                                                     upfront_capex=2e6))

    def test_penalties_reduce_the_score_by_the_stated_factor(self):
        # Share 0.5 is flagged (-0.25) and capped at 0.30 for the computation.
        flagged = score_business(inputs(reachable_share_3y=0.5), DESCRIPTORS)
        capped = unit_economics({**BASE_INPUTS, "reachable_share_3y": 0.30})
        self.assertAlmostEqual(flagged.breakdown["sanity_penalty"], 0.25)
        self.assertAlmostEqual(flagged.breakdown["economics"]["customers_3y"], capped["customers_3y"])
        self.assertAlmostEqual(flagged.score, 100 * flagged.breakdown["base"] * 0.75, places=3)

    def test_invalid_inputs_score_zero(self):
        result = score_business(inputs(monthly_revenue_per_customer=0.0), DESCRIPTORS)
        self.assertEqual(result.score, 0.0)
        self.assertIn("invalid_inputs", [f["code"] for f in result.breakdown["sanity_flags"]])


class CriticScoringTest(unittest.TestCase):
    def test_critic_adjusted_inputs_replace_the_generator_estimates(self):
        adjusted = inputs(cac=800.0, reachable_share_3y=0.02)
        review = CriticReview(order_id="r001-01", verdict="weakened", killer_risks=["incumbent", "regulation"],
                              adjusted_inputs=adjusted, notes="CAC doubled")
        result = score_business(inputs(), DESCRIPTORS, review)
        self.assertEqual(result.breakdown["inputs_source"], "critic")
        self.assertEqual(result.breakdown["inputs_used"]["cac"], 800.0)
        self.assertEqual(result.breakdown["inputs_original"]["cac"], 400.0)
        # weakened 0.15 + 2 risks * 0.05 = 0.25
        self.assertAlmostEqual(result.breakdown["critic_penalty"], 0.25)
        econ = unit_economics(result.breakdown["inputs_used"])
        self.assertAlmostEqual(econ["ltv_cac"], 2.0)  # 1600 / 800
        self.assertAlmostEqual(result.score, 100 * result.breakdown["base"] * 0.75, places=3)
        self.assertLess(result.score, score_business(inputs(), DESCRIPTORS).score)

    def test_review_without_adjusted_inputs_only_penalises(self):
        review = CriticReview(order_id="r001-01", verdict="refuted", killer_risks=["a", "b", "c", "d"])
        result = score_business(inputs(), DESCRIPTORS, review)
        self.assertEqual(result.breakdown["inputs_source"], "generator")
        self.assertAlmostEqual(result.breakdown["critic_penalty"], 0.4 + 0.15)
        self.assertAlmostEqual(result.score, 73.4934 * 0.45, places=2)

    def test_unknown_verdict_counts_as_weakened(self):
        self.assertAlmostEqual(critic_penalty(CriticReview(order_id="x", verdict="Hmm")), 0.15)
        self.assertAlmostEqual(critic_penalty(CriticReview(order_id="x", verdict="SURVIVES")), 0.0)
        self.assertEqual(critic_penalty(None), 0.0)


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = parsed.model_dump_json()


class _FakeClient:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []
        self.models = self

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append({"model": model, "contents": contents, "schema": config.response_schema})
        return _FakeResponse(self.parsed)


class BusinessEvaluateTest(unittest.TestCase):
    def prepared(self, order_id="r001-01"):
        return PreparedCandidate(order={"order_id": order_id, "target": {}}, candidate=candidate(order_id),
                                 descriptors=dict(DESCRIPTORS), concept={})

    def test_mock_critic_is_applied(self):
        results = BusinessProfile().evaluate([self.prepared()], EvaluationContext(query="q", mock=True))
        self.assertTrue(results[0].breakdown["critic_used"])
        self.assertEqual(results[0].breakdown["inputs_source"], "critic")
        self.assertAlmostEqual(results[0].breakdown["inputs_used"]["cac"], 520.0)

    def test_without_critic(self):
        results = BusinessProfile().evaluate([self.prepared()], EvaluationContext(query="q", mock=True, use_critic=False))
        self.assertFalse(results[0].breakdown["critic_used"])
        self.assertAlmostEqual(results[0].score, 73.4934, places=3)

    def test_live_critic_call_uses_the_critic_stage(self):
        review = CriticReview(order_id="r001-01", verdict="survives", killer_risks=[])
        client = _FakeClient(CriticBatch(reviews=[review, CriticReview(order_id="zzz", verdict="refuted")]))
        with mock.patch.dict(os.environ, {"VECTORNAUT_MODEL_CRITIC": "critic-model"}):
            results = BusinessProfile().evaluate(
                [self.prepared()], EvaluationContext(query="Reduce food waste", mock=False, critic_client=client))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["model"], "critic-model")
        self.assertIs(client.calls[0]["schema"], CriticBatch)
        self.assertIn("[r001-01] Shared cold room", client.calls[0]["contents"])
        self.assertIn("Reduce food waste", client.calls[0]["contents"])
        self.assertEqual(results[0].breakdown["critic_verdict"], "survives")


class BusinessCandidateCheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.archive = Archive("business", BUSINESS_SPACE, path=os.path.join(tmp.name, "a.json"))
        self.profile = BusinessProfile()

    def check(self, candidates, orders):
        return check_batch(self.profile, orders, BusinessCandidateBatch(candidates=candidates), self.archive)

    def test_statuses(self):
        orders = [{"order_id": f"r001-0{i}", "strategy": "explore", "target": dict(DESCRIPTORS),
                   "target_key": BUSINESS_SPACE.cell_key(DESCRIPTORS)} for i in range(1, 6)]
        bad_desc = {**DESCRIPTORS, "revenue_model": "donations"}
        result = self.check([
            candidate("r001-01", "Good one"),
            candidate("r001-02", "Bad cell", descriptors=bad_desc),
            candidate("r001-03", "No inputs", inputs=None),
            BusinessCandidate(order_id="r001-04", target_feasible=False, infeasibility_reason="law forbids it"),
            candidate("r001-01", "Second for same order"),
            candidate("r009-09", "Unknown order"),
            candidate("r001-05", "good one"),  # same title as r001-01 after normalisation
        ], orders)
        statuses = [item.status for item in result.items]
        self.assertEqual(statuses, [OK, INVALID, REJECTED, TARGET_INFEASIBLE, REJECTED])
        self.assertIn("revenue_model: 'donations' is not an allowed value", "; ".join(result.items[1].issues))
        self.assertIn("inputs are missing", "; ".join(result.items[2].issues))
        self.assertEqual(result.items[3].issues, ["law forbids it"])
        self.assertIn("duplicate", "; ".join(result.items[4].issues))
        self.assertEqual(len(result.unmatched), 2)

    def test_missing_candidate(self):
        orders = [{"order_id": "r001-01", "strategy": "seed", "target": {}, "target_key": ""}]
        result = self.check([], orders)
        self.assertEqual(result.items[0].status, "missing")


if __name__ == "__main__":
    unittest.main()
