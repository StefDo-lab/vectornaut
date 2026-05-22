# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import re
from datetime import datetime
from pydantic import BaseModel, Field
from google.genai import types
from .config import get_client, MinerOutput, AuditorOutput

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
        os.makedirs("generated_tests", exist_ok=True)
        test_script_path = os.path.abspath(os.path.join("generated_tests", f"test_solver_{slug}_{timestamp}.py"))
        test_output_path = os.path.abspath(os.path.join("generated_tests", f"results_test_{slug}_{timestamp}.json"))
        
        # Aktuelle Parameter des Auditors ermitteln
        audited_params = auditor_output.audited_parameters_dict.copy()
        audited_params["simulation_coefficient"] = auditor_output.simulation_coefficient
        
        # Generierungs-Prompt entwerfen
        prompt = f"""
        Schreibe ein eigenständiges Python-Testskript unter Verwendung des `unittest` Moduls, um das simulierte physikalische System und dessen Solver-Skript zu validieren.
        
        Das Solver-Skript befindet sich an dem Pfad, der über den CLI-Parameter `--solver` übergeben wird.
        Das Solver-Skript akzeptiert folgende Argumente:
          --params: Pfad zu einer JSON-Datei mit Eingangsparametern.
          --output: Pfad, an den das Skript eine JSON-Datei mit den Ergebnissen schreibt.
          --plot: Pfad, an den das Skript ein PNG-Diagramm speichert.

        Das Testskript MUSS folgende CLI-Parameter per argparse akzeptieren:
          --solver: Pfad zum zu testenden Python-Solver-Skript.
          --params: Pfad zur JSON-Datei mit den nominellen Eingangsparametern (wird genutzt, um Standardwerte zu ermitteln).
          --output: Pfad, an den das Testskript eine JSON-Datei mit dem Testergebnis schreiben MUSS.

        ### Zu testendes System:
        Design Name: {miner_output.design_name}
        Physikalische Domäne: {miner_output.domain}
        Mechanismus: {miner_output.physical_mechanism}
        DGL (Governing Equation): {miner_output.governing_equation}
        Randbedingungen: {miner_output.boundary_conditions}
        Nominelle Parameter: {json.dumps(audited_params)}
        Simulationskoeffizient: {auditor_output.simulation_coefficient}

        ### ANFORDERUNGEN AN DAS TESTSKRIPT:
        1. STRUKTUR:
           Verwende das standardmäßige Python `unittest` Modul. Definiere eine Testklasse, z. B. `class SolverPhysicalValidation(unittest.TestCase)`.
           Am Ende des Skripts (im `if __name__ == "__main__":` Block) MUSS die Testsuite ausgeführt werden und das Ergebnis in eine strukturierte JSON-Datei an `--output` geschrieben werden.
           
           Die JSON-Datei MUSS exakt folgende Struktur haben:
           {{
             "success": bool (True, wenn ALLE Tests bestanden wurden, andernfalls False),
             "total_run": int (Anzahl der ausgeführten Tests),
             "total_failures": int,
             "total_errors": int,
             "test_results": [
               {{
                 "name": str (Name der Testmethode, z.B. "test_nominal_run"),
                 "passed": bool (True/False),
                 "message": str (Fehlermeldung bei Fehlschlag oder leerer String "")
               }},
               ...
             ]
           }}

        2. IMPLEMENTIERUNG DER TESTMETHODEN (Mindestens 3 Testfälle):
           
           - Test 1: `test_nominal_run`
             Führe das Solver-Skript per `subprocess.run` mit den nominellen Parametern aus.
             Verifiziere:
               * Der Exit-Code ist 0.
               * Die Ausgabedatei (JSON) wird erfolgreich erzeugt.
               * Die JSON enthält `"success": true`.
               * Alle Pflichtfelder (z. B. `performance_gain_pct`, `solution_primary`, `primary_metric_value`) sind vorhanden und enthalten numerische Werte.

           - Test 2: `test_parameter_limits_and_safety`
             Wähle ein oder zwei Hauptparameter aus den nominellen Parametern (z.B. slip_length, viscosity, conductivity, thickness) und setze sie auf extreme Werte (sehr klein, z.B. 0.0 oder 1e-8, oder sehr groß).
             Führe das Solver-Skript mit diesen geänderten Parametern aus.
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

        Verwende für die Subprocess-Aufrufe `sys.executable` als Python-Interpreter, um Kompatibilität zu gewährleisten. Erstelle temporäre JSON- und PNG-Dateien für die Testdurchläufe und lösche sie nach jedem Testlauf (im `tearDown` oder per try-finally).
        Schreibe sauberen, robusten Python 3.13 Code.
        """

        correction_iteration = 0
        max_iterations = 3
        current_prompt = prompt
        error_msg = ""
        
        while correction_iteration < max_iterations:
            print(f"[*] Generating test script iteration {correction_iteration + 1}...")
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=current_prompt,
                config=types.GenerateContentConfig(
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
            
            # Ausführen des Testskripts
            print(f"[*] Executing generated test script at: {test_script_path}")
            result = subprocess.run(
                [sys.executable, test_script_path, "--solver", solver_script_path, "--params", params_json_path, "--output", test_output_path],
                capture_output=True,
                text=True
            )
            
            # Überprüfen ob das Testskript selbst fehlerfrei durchgelaufen ist (unabhängig davon ob Tests fehlschlagen)
            # Das Testskript sollte exit code 0 haben, auch wenn Tests fehlschlagen, da unittest.main(exit=False) verwendet wird,
            # oder zumindest die Ausgabedatei geschrieben wurde.
            if os.path.exists(test_output_path):
                try:
                    with open(test_output_path, "r", encoding="utf-8") as rf:
                        test_results_data = json.load(rf)
                    test_results_data["test_script_path"] = test_script_path
                    test_results_data["test_output_path"] = test_output_path
                    print(f"[+] Test script execution succeeded on iteration {correction_iteration + 1}!")
                    return test_results_data
                except Exception as read_err:
                    error_msg = f"Failed to read test results JSON: {str(read_err)}"
                    print(f"[-] Failed to read test output JSON: {read_err}")
            else:
                error_msg = result.stderr or result.stdout or "Test script did not produce output JSON"
                print(f"[-] Test script crashed or failed to write JSON on iteration {correction_iteration + 1}!")
                print(f"[-] Error: {error_msg}")
            
            # Vorbereitung der Korrekturschleife
            correction_iteration += 1
            if correction_iteration < max_iterations:
                print(f"[*] Initiating test-correction. Sending error details back to Gemini...")
                current_prompt = f"""
                Das zuvor generierte Python-Testskript ist beim Ausführen abgestürzt oder hat das erwartete JSON nicht erzeugt!
                
                ### Zuvor generierter Testcode:
                ```python
                {code}
                ```
                
                ### Fehlermeldung (Traceback / Stderr / Stdout):
                ```text
                {error_msg}
                ```
                
                ### Aufgabe:
                Analysiere den Fehler, korrigiere den Code und liefere ein repariertes, vollständig lauffähiges Testskript zurück.
                Halte dich strikt an die CLI-Parameter (--solver, --params, --output) und stelle sicher, dass am Ende des Skripts die Ergebnisse korrekt als JSON exportiert werden.
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
