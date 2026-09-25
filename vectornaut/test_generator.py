# -*- coding: utf-8 -*-
import os
import json
import re
from datetime import datetime
from pydantic import BaseModel, Field
from google.genai import types
from .config import get_client, get_model_name, get_thinking_config, MinerOutput, AuditorOutput
from .storage import data_path
from .sandbox import (
    check_generated_code,
    copy_if_exists,
    format_rejection,
    run_generated_script,
    sandbox_workdir,
    script_timeout_seconds,
    SANDBOX_RULES_TEXT,
)

# Trusted runner that imports the generated tests and derives pass/fail from unittest.
_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_test_runner.py")


def validation_timeout_seconds() -> int:
    """Wall-clock budget for the whole test script (it runs the solver several times)."""
    try:
        return max(1, int(os.environ["VECTORNAUT_TEST_SCRIPT_TIMEOUT_SECONDS"]))
    except (KeyError, ValueError):
        return 5 * script_timeout_seconds()

class GeneratedTestScriptResponse(BaseModel):
    explanation: str = Field(description="Kurze Erklärung der gewählten Validierungstests und physikalischen Invarianten.")
    code: str = Field(description="Der vollständige, lauffähige Python-Testcode. Keine Markdown-Fences drumherum, reiner Code.")

class TestScriptGenerator:
    def __init__(self, client=None):
        self.client = client or get_client()

    def generate_and_execute_tests(
        self,
        miner_output: MinerOutput,
        auditor_output: AuditorOutput,
        solver_script_path: str,
        params_json_path: str
    ) -> dict:
        """
        Generiert ein Python-Validierungs-Testskript, führt es aus und korrigiert es bei Fehlern selbstständig.
        """
        design_name = miner_output.design_name or "unknown_design"
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Pfade festlegen
        generated_tests_dir = data_path("generated_tests")
        os.makedirs(generated_tests_dir, exist_ok=True)
        test_script_path = os.path.abspath(os.path.join(generated_tests_dir, f"test_solver_{slug}_{timestamp}.py"))
        test_output_path = os.path.abspath(os.path.join(generated_tests_dir, f"results_test_{slug}_{timestamp}.json"))
        
        # Aktuelle Parameter des Auditors ermitteln
        audited_params = auditor_output.audited_parameters_dict.copy()
        audited_params["simulation_coefficient"] = auditor_output.simulation_coefficient
        
        # Generierungs-Prompt entwerfen
        prompt = f"""
        Schreibe ein Python-Testmodul unter Verwendung des `unittest` Moduls, um das simulierte physikalische System und dessen Solver-Skript zu validieren.

        ### AUSFÜHRUNGSMODELL (verbindlich):
        Das Testmodul wird NICHT direkt ausgeführt. Ein vertrauenswürdiger Test-Runner importiert es in einer Sandbox,
        führt alle `unittest.TestCase`-Klassen darin aus und bestimmt Bestanden/Fehlgeschlagen selbst aus dem unittest-Ergebnis.
        Deshalb:
          - KEIN argparse, KEIN `if __name__ == "__main__":`-Block, KEINE eigene JSON-Ergebnisdatei.
          - Das Testmodul startet KEINE Prozesse selbst (kein subprocess, kein os.system). Der Solver wird ausschließlich über das
            vom Runner bereitgestellte Modul `solver_harness` aufgerufen:

              from solver_harness import run_solver, nominal_params, is_finite_number

              nominal_params() -> dict
                  Frische Kopie der nominellen Parameter (siehe unten).
              run_solver(params: dict | None = None, timeout: float | None = None) -> SolverRun
                  Führt das Solver-Skript mit den übergebenen Parametern aus (None = nominelle Parameter).
                  Rückgabe: benanntes Tupel SolverRun(returncode: int, output: dict | None, stderr: str);
                  `output` ist das geparste Ergebnis-JSON des Solvers (oder None, falls keines geschrieben wurde).
                  Temporäre Dateien verwaltet der Runner.
              is_finite_number(x) -> bool

        Das Ergebnis-JSON des Solvers enthält mindestens: success, performance_gain_pct, relative_error, sample_points,
        solution_primary, solution_reference, primary_metric_value, reference_metric_value.

        ### Zu testendes System:
        Design Name: {miner_output.design_name}
        Physikalische Domäne: {miner_output.domain}
        Mechanismus: {miner_output.physical_mechanism}
        DGL (Governing Equation): {miner_output.governing_equation}
        Randbedingungen: {miner_output.boundary_conditions}
        Nominelle Parameter: {json.dumps(audited_params)}
        Simulationskoeffizient: {auditor_output.simulation_coefficient}

        ### ANFORDERUNGEN AN DAS TESTMODUL:
        1. STRUKTUR:
           Definiere eine Testklasse, z. B. `class SolverPhysicalValidation(unittest.TestCase)`, mit mindestens 3 Testmethoden.
           Fehlschläge werden über normale unittest-Assertions ausgedrückt (self.assertEqual, self.assertTrue, ...).

        2. IMPLEMENTIERUNG DER TESTMETHODEN (Mindestens 3 Testfälle):

           - Test 1: `test_nominal_run`
             Führe den Solver per `run_solver(nominal_params())` aus.
             Verifiziere:
               * Der Exit-Code ist 0.
               * `output` ist nicht None.
               * `output["success"]` ist True.
               * Alle Pflichtfelder (z. B. `performance_gain_pct`, `solution_primary`, `primary_metric_value`) sind vorhanden und enthalten numerische Werte.

           - Test 2: `test_parameter_limits_and_safety`
             Wähle ein oder zwei Hauptparameter aus den nominellen Parametern (z.B. slip_length, viscosity, conductivity, thickness) und setze sie auf extreme Werte (sehr klein, z.B. 0.0 oder 1e-8, oder sehr groß).
             Führe den Solver mit diesen geänderten Parametern aus.
             Verifiziere:
               * Das Skript stürzt nicht ab (Exit-Code 0).
               * Die Ergebnisse enthalten keine NaN- oder unendlichen Werte.

           - Test 3: `test_physical_invariants`
             Überprüfe physikalische Trends und logische Invarianten basierend auf dem physikalischen Mechanismus:
               * Falls Drag Reduction / Slip-System:
                 - Wenn slip_length = 0 (oder sehr nah an 0), sollte die bionische Metrik (Wandschubspannung) der Referenzmetrik entsprechen, d.h. der relative Fehler zur Referenzlösung ist extrem klein und `performance_gain_pct` ist nahe 0 (oder 0).
                 - Wenn slip_length erhöht wird, sollte die Reibung abnehmen, d.h. `performance_gain_pct` sollte steigen (oder zumindest nicht sinken).
               * Falls Thermische Isolation / Heat Shield:
                 - Alle Temperaturwerte in `solution_primary` müssen innerhalb der Grenzen `[T_cold, T_hot]` (bzw. Umgebungstemperaturen) liegen. Ein Überschreiten ist physikalisch unmöglich (Maximumprinzip).
                 - Wenn die thermische Leitfähigkeit des bionischen Materials gesenkt wird, sollte der Wärmestrom sinken (Isolationswirkung steigt).
               * Wähle die Invarianten passend für: "{miner_output.domain}"!

        3. SANDBOX:
           {SANDBOX_RULES_TEXT}
           Der gesamte Testlauf hat ein Zeitbudget von {validation_timeout_seconds()} Sekunden; halte die Zahl der Solver-Aufrufe klein.
           Falls du selbst Dateien öffnest, verwende explizit `encoding="utf-8"`.

        Schreibe sauberen, robusten Python 3.13 Code.
        """

        correction_iteration = 0
        max_iterations = 3
        current_prompt = prompt
        error_msg = ""
        
        while correction_iteration < max_iterations:
            print(f"[*] Generating test script iteration {correction_iteration + 1}...")
            response = self.client.models.generate_content(
                model=get_model_name("test_generator"),
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=get_thinking_config("test_generator"),
                    response_mime_type="application/json",
                    response_schema=GeneratedTestScriptResponse,
                )
            )
            
            parsed: GeneratedTestScriptResponse = response.parsed
            code = parsed.code
            
            # Markdown fences entfernen falls vorhanden
            if code.startswith("```python"):
                code = code.replace("```python", "", 1)
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()

            # Skript auf Festplatte schreiben
            with open(test_script_path, "w", encoding="utf-8") as tf:
                tf.write(code)

            violations = check_generated_code(code)
            if violations:
                error_msg = format_rejection(violations)
                headline = "Das zuvor generierte Testmodul wurde von der Sandbox-Vorprüfung abgelehnt und nicht ausgeführt!"
                print(f"[-] Test script rejected by sandbox pre-check on iteration {correction_iteration + 1}!")
                print(f"[-] {error_msg}")
            else:
                print(f"[*] Executing generated test script at: {test_script_path}")
                test_results_data, error_msg = self._execute(
                    test_script_path, solver_script_path, params_json_path, test_output_path
                )
                if test_results_data is not None:
                    print(f"[+] Test script execution succeeded on iteration {correction_iteration + 1}!")
                    return test_results_data
                headline = "Das zuvor generierte Testmodul konnte nicht importiert/ausgeführt werden oder enthielt keine Tests!"
                print(f"[-] Test script could not be run on iteration {correction_iteration + 1}!")
                print(f"[-] Error: {error_msg}")

            # Vorbereitung der Korrekturschleife
            correction_iteration += 1
            if correction_iteration < max_iterations:
                print(f"[*] Initiating test-correction. Sending error details back to Gemini...")
                current_prompt = f"""
                {headline}

                ### Zuvor generierter Testcode:
                ```python
                {code}
                ```

                ### Fehlermeldung (Traceback / Stderr / Prüfergebnis):
                ```text
                {error_msg}
                ```

                ### Aufgabe:
                Analysiere den Fehler, korrigiere den Code und liefere ein repariertes Testmodul zurück.
                Vertrag: ein unittest-Modul ohne argparse, ohne __main__-Block und ohne eigene Ergebnisdatei; der Solver wird nur über
                `from solver_harness import run_solver, nominal_params, is_finite_number` aufgerufen
                (run_solver(params) -> SolverRun(returncode, output, stderr)). Der Runner wertet die unittest-Ergebnisse selbst aus.
                {SANDBOX_RULES_TEXT}
                """

        # Falls alle Iterationen fehlgeschlagen sind
        return {
            "success": False,
            "test_script_path": test_script_path,
            "test_output_path": test_output_path,
            "total_run": 0,
            "total_failures": 0,
            "total_errors": 1,
            "test_results": [
                {
                    "name": "test_generation_failure",
                    "passed": False,
                    "message": f"Test script generation or compilation failed: {error_msg}"
                }
            ]
        }


    @staticmethod
    def _execute(test_script_path: str, solver_script_path: str, params_json_path: str, test_output_path: str):
        """
        Run the generated tests through the trusted runner in the sandbox. Returns
        (results, error_msg); results is None when the tests could not be run at all
        (import error, no tests, timeout, no summary), which triggers self-correction.
        Pass/fail comes from the unittest result collected by the runner.
        """
        with sandbox_workdir("vectornaut_tests_") as workdir:
            copy_if_exists(_RUNNER_PATH, os.path.join(workdir, "_vectornaut_test_runner.py"))
            copy_if_exists(test_script_path, os.path.join(workdir, "generated_tests.py"))
            copy_if_exists(solver_script_path, os.path.join(workdir, "solver.py"))
            copy_if_exists(params_json_path, os.path.join(workdir, "params.json"))
            summary_name = f"summary_{os.urandom(8).hex()}.json"
            run = run_generated_script(
                ["_vectornaut_test_runner.py",
                 "--tests", "generated_tests.py",
                 "--solver", "solver.py",
                 "--params", "params.json",
                 "--summary", summary_name,
                 "--solver-timeout", str(script_timeout_seconds())],
                timeout=validation_timeout_seconds(),
                workdir=workdir,
            )
            summary = None
            summary_path = os.path.join(workdir, summary_name)
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as rf:
                        summary = json.load(rf)
                except Exception as read_err:
                    return None, f"Failed to read the test runner summary: {read_err}"

        if run.timed_out:
            return None, (
                f"Test script timed out after {run.duration_s:.0f} seconds (process group killed). "
                "Reduce the number or size of solver runs."
            )
        if not isinstance(summary, dict):
            return None, run.error_text() or f"Test runner exited with code {run.returncode} without a summary."
        if not summary.get("collected"):
            return None, summary.get("import_error") or run.error_text() or "The test module could not be loaded."
        expected_code = 0 if summary.get("success") else 1
        if run.returncode != expected_code:
            return None, (
                f"Test runner exit code {run.returncode} does not match the unittest result "
                f"(success={summary.get('success')}); the test module must not exit the interpreter.\n{run.error_text(2000)}"
            )

        test_results_data = {
            "success": bool(summary.get("success")),
            "total_run": int(summary.get("total_run", 0)),
            "total_failures": int(summary.get("total_failures", 0)),
            "total_errors": int(summary.get("total_errors", 0)),
            "total_skipped": int(summary.get("total_skipped", 0)),
            "test_results": summary.get("test_results", []),
            "result_source": "unittest runner (harness)",
            "test_script_path": test_script_path,
            "test_output_path": test_output_path,
        }
        with open(test_output_path, "w", encoding="utf-8") as wf:
            json.dump(test_results_data, wf, indent=2, ensure_ascii=False)
        return test_results_data, ""
