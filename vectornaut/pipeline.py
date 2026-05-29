# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from vectornaut.config import AuditorOutput, MinerOutput
from vectornaut.runtime_guards import run_limits, validate_simulator_output
from vectornaut.validator import validate_run_output


class _PipelineValue:
    def __init__(self, **values: Any):
        self.__dict__.update(values)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _AuditorSolverOverride:
    def __init__(self, original: Any, solver_method: str):
        self._original = original
        self.solver_method = solver_method

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def model_dump(self) -> Dict[str, Any]:
        if hasattr(self._original, "model_dump"):
            data = self._original.model_dump()
        else:
            data = dict(getattr(self._original, "__dict__", {}))
        data["solver_method"] = self.solver_method
        return data


class PipelineRunRequest(BaseModel):
    query: str
    epochs: int = 200
    is_mock: bool = False
    override_parameters: Optional[Dict[str, float]] = None
    previous_miner_output: Optional[Any] = None
    max_optimization_rounds: int = 3


class PipelineRunner:
    def __init__(
        self,
        miner: Optional[Any] = None,
        formulator: Optional[Any] = None,
        auditor: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        synthesizer: Optional[Any] = None,
        solver: Optional[Any] = None,
    ):
        if miner is None:
            from .miner import Miner
            miner = Miner()
        if formulator is None:
            from .formulator import ModelFormulator
            formulator = ModelFormulator()
        if auditor is None:
            from .auditor import Auditor
            auditor = Auditor()
        if optimizer is None:
            from .optimizer import Optimizer
            optimizer = Optimizer()
        if synthesizer is None:
            from .synthesizer import Synthesizer
            synthesizer = Synthesizer()
        self.miner = miner
        self.formulator = formulator
        self.auditor = auditor
        self.optimizer = optimizer
        self.synthesizer = synthesizer
        self.solver = solver

    def _solve(self, miner_output: Any, auditor_output: Any, epochs: int, effective_mock: bool) -> Any:
        if self.solver is not None:
            return self.solver(miner_output=miner_output, auditor_output=auditor_output, epochs=epochs)

        try:
            from .solver_dispatcher import dispatch_and_solve
        except ImportError:
            if not effective_mock:
                raise
            print("[*] Solver dependencies missing; using lightweight mock solver.")
            return _PipelineValue(
                solver_method="mock",
                epochs_trained=0,
                final_loss=0.0,
                loss_history=[],
                performance_gain_pct=0.0,
                relative_error=0.0,
                sample_points=[0.0, 1.0],
                solution_primary=[0.0, 1.0],
                solution_reference=[0.0, 1.0],
                primary_metric_value=1.0,
                reference_metric_value=1.0,
                custom_plot_url=None,
                validation_passed=None,
                validation_report=None,
                validation_tests=None,
            )

        return dispatch_and_solve(
            miner_output=miner_output,
            auditor_output=auditor_output,
            epochs=epochs,
        )

    def _validation_for(
        self,
        miner_output: Any,
        auditor_output: Any,
        sim_output: Any,
        optimization_history: List[Dict[str, Any]],
    ) -> Any:
        return validate_run_output(
            miner_output=miner_output,
            auditor_output={
                **auditor_output.model_dump(),
                "audited_parameters_dict": auditor_output.audited_parameters_dict,
                "dimensionless_numbers_dict": auditor_output.dimensionless_numbers_dict,
            },
            simulator_output=sim_output,
            optimization_history=optimization_history,
        )

    def _fallback_solver_methods(self, miner_output: Any, auditor_output: Any) -> List[str]:
        current = str(getattr(auditor_output, "solver_method", "") or "").lower()
        independent_vars = getattr(miner_output, "independent_variables", []) or []
        is_2d = len(independent_vars) == 2
        candidates = ["fdm"] if is_2d else ["analytical", "scipy"]
        return [method for method in candidates if method != current]

    def _try_solver_fallbacks(
        self,
        miner_output: Any,
        auditor_output: Any,
        sim_output: Any,
        validation_result: Any,
        optimization_history: List[Dict[str, Any]],
        epochs: int,
        effective_mock: bool,
    ) -> Tuple[Any, Any, Any, List[Dict[str, Any]]]:
        if validation_result.recommended_action != "rerun_solver":
            return auditor_output, sim_output, validation_result, []

        fallback_attempts = []
        best_auditor = auditor_output
        best_sim = sim_output
        best_validation = validation_result

        for method in self._fallback_solver_methods(miner_output, auditor_output):
            print(f"[*] Validator requested solver fallback. Trying {method.upper()}...")
            fallback_auditor = _AuditorSolverOverride(auditor_output, method)
            attempt: Dict[str, Any] = {"solver_method": method, "status": "failed"}

            try:
                candidate_sim = self._solve(
                    miner_output=miner_output,
                    auditor_output=fallback_auditor,
                    epochs=epochs,
                    effective_mock=effective_mock,
                )
                validate_simulator_output(candidate_sim)
                candidate_validation = self._validation_for(
                    miner_output=miner_output,
                    auditor_output=fallback_auditor,
                    sim_output=candidate_sim,
                    optimization_history=optimization_history,
                )
                attempt.update({
                    "status": candidate_validation.status,
                    "reliability": candidate_validation.reliability,
                    "score": candidate_validation.score,
                    "recommended_action": candidate_validation.recommended_action,
                })

                current_score = float(getattr(best_validation, "score", 0.0) or 0.0)
                candidate_score = float(candidate_validation.score or 0.0)
                improves = candidate_validation.status == "pass" or (
                    candidate_validation.status != "fail" and candidate_score > current_score
                )
                if improves:
                    best_auditor = fallback_auditor
                    best_sim = candidate_sim
                    best_validation = candidate_validation
                    print(f"[+] Solver fallback accepted: {method.upper()} ({candidate_validation.status}, score={candidate_validation.score})")
                    if candidate_validation.status == "pass":
                        fallback_attempts.append(attempt)
                        break
            except Exception as fallback_err:
                attempt["error"] = str(fallback_err)
                print(f"[-] Solver fallback {method.upper()} failed: {fallback_err}")

            fallback_attempts.append(attempt)

        return best_auditor, best_sim, best_validation, fallback_attempts

    def run(self, req: PipelineRunRequest) -> Dict[str, Any]:
        api_key = os.environ.get("GEMINI_API_KEY")
        effective_mock = req.is_mock
        if not api_key and not effective_mock:
            effective_mock = True
        safe_epochs, max_opt_rounds = run_limits(req.epochs, req.max_optimization_rounds)

        failed_concepts = []
        miner_output = None
        auditor_output = None
        sim_output = None
        validation_result = None
        optimization_history = []

        max_concept_attempts = 3
        concept_failed = False
        concept_attempt = 1

        for concept_attempt in range(1, max_concept_attempts + 1):
            print(f"\n[*] Starting Concept Attempt {concept_attempt}...")
            concept_failed = False
            validation_result = None

            if req.previous_miner_output and concept_attempt == 1:
                miner_output = req.previous_miner_output
                print(f"[*] Reusing cached concept: {miner_output.design_name}")
            else:
                if effective_mock:
                    concept = self.miner.mock_mine_design(req.query)
                    if concept_attempt > 1:
                        concept.design_name = f"Alternative {concept.design_name} (Attempt {concept_attempt})"
                        concept.inspiration_source = "Nelumbo nucifera (Lotus leaf)"
                        concept.physical_mechanism = "Superhydrophobic surface structures reduce the wetted contact area."
                    miner_output = self.formulator.mock_formulate_model(req.query, concept)
                else:
                    concept = self.miner.mine_design(req.query, failed_concepts=failed_concepts)
                    miner_output = self.formulator.formulate_model(req.query, concept)

            print(f"[+] Concept Mined & Formulated: {miner_output.design_name}")

            current_overrides = req.override_parameters.copy() if req.override_parameters else {}
            optimization_history = []

            for round_idx in range(1, max_opt_rounds + 1):
                print(f"\n[*] Running Optimization Round {round_idx} for {miner_output.design_name}...")
                active_overrides = current_overrides.copy()

                if effective_mock:
                    auditor_output = self.auditor.mock_audit_design(
                        miner_output,
                        override_parameters=active_overrides,
                        user_query=req.query,
                    )
                else:
                    auditor_output = self.auditor.audit_design(
                        miner_output,
                        override_parameters=active_overrides,
                        user_query=req.query,
                    )

                if not auditor_output.audit_passed:
                    print(f"[-] Auditor failed at round {round_idx}: {auditor_output.audit_notes}")
                    failed_concepts.append({
                        "design_name": miner_output.design_name,
                        "inspiration_source": miner_output.inspiration_source,
                        "reason": f"Audit failed: {auditor_output.audit_notes}",
                    })
                    concept_failed = True
                    break

                try:
                    sim_output = self._solve(
                        miner_output=miner_output,
                        auditor_output=auditor_output,
                        epochs=safe_epochs,
                        effective_mock=effective_mock,
                    )
                    validate_simulator_output(sim_output)
                except Exception as solve_err:
                    print(f"[-] Solver failed to solve equations: {solve_err}")
                    failed_concepts.append({
                        "design_name": miner_output.design_name,
                        "inspiration_source": miner_output.inspiration_source,
                        "reason": f"Simulation solver error: {solve_err}",
                    })
                    concept_failed = True
                    break

                round_data = {
                    "round": round_idx,
                    "parameters": auditor_output.audited_parameters_dict,
                    "simulation_coefficient": auditor_output.simulation_coefficient,
                    "simulator": sim_output.model_dump(),
                    "optimizer_reasoning": "",
                }
                optimization_history.append(round_data)
                validation_result = self._validation_for(
                    miner_output=miner_output,
                    auditor_output=auditor_output,
                    sim_output=sim_output,
                    optimization_history=optimization_history,
                )
                auditor_output, sim_output, validation_result, fallback_attempts = self._try_solver_fallbacks(
                    miner_output=miner_output,
                    auditor_output=auditor_output,
                    sim_output=sim_output,
                    validation_result=validation_result,
                    optimization_history=optimization_history,
                    epochs=safe_epochs,
                    effective_mock=effective_mock,
                )
                if fallback_attempts:
                    round_data["solver_fallbacks"] = fallback_attempts
                    round_data["simulator"] = sim_output.model_dump()
                    round_data["parameters"] = auditor_output.audited_parameters_dict
                round_data["validation"] = validation_result.model_dump()

                if validation_result.status == "fail":
                    failed_detail = "; ".join(
                        check.detail
                        for check in validation_result.checks
                        if check.severity == "error" and not check.passed
                    )
                    print(f"[-] Validator rejected result: {failed_detail}")
                    failed_concepts.append({
                        "design_name": miner_output.design_name,
                        "inspiration_source": miner_output.inspiration_source,
                        "reason": f"Validator failed: {failed_detail}",
                    })
                    concept_failed = True
                    break

                if round_idx < max_opt_rounds:
                    print(f"[*] Evaluating round {round_idx} with Optimizer...")
                    if effective_mock:
                        opt_decision = self.optimizer.mock_optimize(miner_output, optimization_history)
                    else:
                        opt_decision = self.optimizer.optimize(miner_output, optimization_history)

                    round_data["optimizer_reasoning"] = opt_decision.reasoning
                    print(f"[+] Optimizer reasoning: {opt_decision.reasoning}")

                    lower_reasoning = opt_decision.reasoning.lower()
                    if "kollaps" in lower_reasoning or "instabil" in lower_reasoning or "versagen" in lower_reasoning:
                        print("[-] Optimizer detected structural collapse/instability. Discarding concept.")
                        failed_concepts.append({
                            "design_name": miner_output.design_name,
                            "inspiration_source": miner_output.inspiration_source,
                            "reason": f"Optimizer rejected: {opt_decision.reasoning}",
                        })
                        concept_failed = True
                        break

                    if not opt_decision.continue_optimization:
                        print("[+] Optimizer decided to terminate optimization loop (convergence or target reached).")
                        break

                    print("[*] Suggested adjustments:")
                    for adj in opt_decision.adjustments:
                        print(f"    - {adj.name} = {adj.value}")
                        current_overrides[adj.name] = adj.value
                else:
                    print("[*] Reached max optimization rounds limit for this concept.")

            if not concept_failed:
                print(f"[+] Concept {miner_output.design_name} successfully optimized!")
                break

            print(f"[-] Concept {miner_output.design_name} failed. Attempting Re-Mining...")

        if concept_failed and concept_attempt == max_concept_attempts:
            raise ValueError(f"All bionic concepts failed validation. Failed history: {failed_concepts}")

        print("[*] Generating Practical & Commercial Synthesis Report...")
        if effective_mock:
            synthesis_report = self.synthesizer.mock_generate_synthesis()
        else:
            synthesis_report = self.synthesizer.generate_synthesis(
                miner_output=miner_output,
                auditor_output=auditor_output,
                simulator_output=sim_output,
                user_query=req.query,
            )
        print("[+] Synthesis report generated successfully.")
        if validation_result is None:
            validation_result = self._validation_for(
                miner_output=miner_output,
                auditor_output=auditor_output,
                sim_output=sim_output,
                optimization_history=optimization_history,
            )

        return {
            "success": True,
            "query": req.query,
            "epochs": safe_epochs,
            "is_mock": effective_mock,
            "miner": miner_output.model_dump(),
            "auditor": {
                **auditor_output.model_dump(),
                "audited_parameters_dict": auditor_output.audited_parameters_dict,
                "dimensionless_numbers_dict": auditor_output.dimensionless_numbers_dict,
            },
            "simulator": sim_output.model_dump(),
            "validation": validation_result.model_dump(),
            "optimization_history": optimization_history,
            "synthesis": synthesis_report.model_dump(),
            "failed_concepts": failed_concepts,
        }

    def compare_solvers(
        self,
        current_run: Dict[str, Any],
        methods: Optional[List[str]] = None,
        epochs: int = 200,
        is_mock: bool = False,
    ) -> Dict[str, Any]:
        miner_output = MinerOutput.model_validate(current_run.get("miner", {}))
        auditor_data = dict(current_run.get("auditor", {}))
        if not isinstance(auditor_data.get("ui_metadata"), dict) or not auditor_data.get("ui_metadata"):
            auditor_data["ui_metadata"] = {
                "domain_name": miner_output.domain or "Scientific Domain",
                "independent_var": {"label": (miner_output.independent_variables or ["x"])[0], "unit": ""},
                "dependent_var": {"label": (miner_output.dependent_variables or ["u"])[0], "unit": ""},
                "primary_metric": {"label": "Primary Metric"},
                "reference_metric": {"label": "Reference Metric"},
                "performance_gain": {"label": "Performance Gain"},
            }
        auditor_output = AuditorOutput.model_validate(auditor_data)
        safe_epochs, _ = run_limits(epochs, 1)
        base_method = str(current_run.get("simulator", {}).get("solver_method") or auditor_output.solver_method).lower()
        candidate_methods = methods or [base_method, *self._fallback_solver_methods(miner_output, auditor_output)]

        results = []
        seen = set()
        for method in candidate_methods:
            normalized_method = str(method or "").lower()
            if not normalized_method or normalized_method in seen:
                continue
            seen.add(normalized_method)

            attempt: Dict[str, Any] = {"solver_method": normalized_method, "status": "failed"}
            try:
                method_auditor = _AuditorSolverOverride(auditor_output, normalized_method)
                sim_output = self._solve(
                    miner_output=miner_output,
                    auditor_output=method_auditor,
                    epochs=safe_epochs,
                    effective_mock=is_mock,
                )
                validate_simulator_output(sim_output)
                validation = self._validation_for(
                    miner_output=miner_output,
                    auditor_output=method_auditor,
                    sim_output=sim_output,
                    optimization_history=[],
                )
                sim_data = sim_output.model_dump()
                attempt.update({
                    "status": validation.status,
                    "reliability": validation.reliability,
                    "score": validation.score,
                    "recommended_action": validation.recommended_action,
                    "relative_error": sim_data.get("relative_error"),
                    "performance_gain_pct": sim_data.get("performance_gain_pct"),
                    "primary_metric_value": sim_data.get("primary_metric_value"),
                    "reference_metric_value": sim_data.get("reference_metric_value"),
                    "validation_passed": sim_data.get("validation_passed"),
                    "validation_tests": sim_data.get("validation_tests"),
                    "script_path": sim_data.get("script_path"),
                    "params_json_path": sim_data.get("params_json_path"),
                    "test_script_path": sim_data.get("test_script_path"),
                    "test_output_path": sim_data.get("test_output_path"),
                    "execution_mode": sim_data.get("execution_mode"),
                    "warnings": validation.warnings[:3],
                })
            except Exception as err:
                attempt["error"] = str(err)
            results.append(attempt)

        successful = [item for item in results if item.get("status") in {"pass", "warn"}]
        best = None
        if successful:
            best = max(successful, key=lambda item: float(item.get("score") or 0.0))

        return {
            "success": True,
            "base_solver": base_method,
            "results": results,
            "best_solver": best.get("solver_method") if best else None,
            "best_status": best.get("status") if best else None,
        }


def run_pipeline(req: PipelineRunRequest) -> Dict[str, Any]:
    return PipelineRunner().run(req)


def compare_solvers(current_run: Dict[str, Any], methods: Optional[List[str]] = None, epochs: int = 200, is_mock: bool = False) -> Dict[str, Any]:
    return PipelineRunner().compare_solvers(current_run=current_run, methods=methods, epochs=epochs, is_mock=is_mock)
