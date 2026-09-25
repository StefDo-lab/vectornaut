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
from .solvers.dynamic_script import (
    MAX_SWEEP_RELATIVE_ERROR,
    _dynamic_objective_contract,
    dynamic_plots_dir,
    solver_output_problems,
)

class GeneratedScriptResponse(BaseModel):
    explanation: str = Field(description="Kurze Erklärung der gewählten numerischen Lösungsmethode für das Skript.")
    code: str = Field(description="Der vollständige, lauffähige Python-Code. Keine Markdown-Fences drumherum, reiner Code.")

class ScriptGenerator:
    def __init__(self, client=None):
        self.client = client or get_client()

    def generate_and_execute(
        self, 
        miner_output: MinerOutput, 
        auditor_output: AuditorOutput, 
        epochs: int = 200,
        plot_png_path: str = None
    ) -> dict:
        """
        Generiert ein Python-Skript für die Simulation, führt es aus und korrigiert es bei Fehlern selbstständig.
        """
        design_name = miner_output.design_name or "unknown_design"
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Pfade festlegen
        generated_dir = data_path("generated_scripts")
        os.makedirs(generated_dir, exist_ok=True)
        script_path = os.path.abspath(os.path.join(generated_dir, f"solver_{slug}_{timestamp}.py"))
        
        # Falls kein Pfad für den Plot übergeben wurde, erstelle einen Standardpfad im Plot-Ordner
        # (static/plots des Repos, per VECTORNAUT_PLOTS_DIR konfigurierbar)
        if not plot_png_path:
            plots_dir = dynamic_plots_dir()
            os.makedirs(plots_dir, exist_ok=True)
            plot_png_path = os.path.abspath(os.path.join(plots_dir, f"plot_{slug}_{timestamp}.png"))
        else:
            plot_png_path = os.path.abspath(plot_png_path)
            os.makedirs(os.path.dirname(plot_png_path), exist_ok=True)

        params_json_path = os.path.abspath(os.path.join(generated_dir, f"params_{slug}_{timestamp}.json"))
        output_json_path = os.path.abspath(os.path.join(generated_dir, f"results_{slug}_{timestamp}.json"))
        
        # Aktuelle Parameter des Auditors speichern
        audited_params = auditor_output.audited_parameters_dict.copy()
        audited_params["simulation_coefficient"] = auditor_output.simulation_coefficient
        audited_params.setdefault("slippage_coefficient", auditor_output.simulation_coefficient)
        audited_params.setdefault("lambda", auditor_output.simulation_coefficient)
        audited_params.setdefault("slip_length", auditor_output.simulation_coefficient)
        
        with open(params_json_path, "w", encoding="utf-8") as pf:
            json.dump(audited_params, pf, indent=4)

        objective_metric = _dynamic_objective_contract(auditor_output)

        # Generierungs-Prompt entwerfen
        prompt = f"""
        Schreibe ein eigenständiges Python-Skript, um das folgende physikalische System numerisch zu lösen.
        
        ### Design-Konzept
        Name: {miner_output.design_name}
        Inspiration: {miner_output.inspiration_source}
        Physikalische Domäne: {miner_output.domain}
        Mechanismus: {miner_output.physical_mechanism}
        
        ### Mathematisches System
        DGL (Governing Equation): {miner_output.governing_equation}
        Randbedingungen (Boundary Conditions): {miner_output.boundary_conditions}
        Unabhängige Variablen: {miner_output.independent_variables}
        Abhängige Variablen: {miner_output.dependent_variables}
        
        ### Parameter (aus der übergebenen JSON)
        Die Parameter sind in der Eingabedatei definiert. Ein wichtiger abgeleiteter Koeffizient ist:
        simulation_coefficient = {auditor_output.simulation_coefficient}
        
        ### VERBINDLICHER METRIC CONTRACT
        Das Skript MUSS diese Bewertungslogik einhalten und darf keine eigene Zielmetrik erfinden:
        {json.dumps(objective_metric, ensure_ascii=False, indent=2)}

        `primary_metric_value`, `reference_metric_value` und `performance_gain_pct` muessen konsistent aus diesem Contract abgeleitet werden.
        Wenn `lower_is_better=true`, ist eine Verringerung der primary_metric gegenueber reference eine Verbesserung.
        Wenn `lower_is_better=false`, ist eine Erhoehung der primary_metric gegenueber reference eine Verbesserung.
        Der Wert in `score_field` ist die einzige Optimierungs-Zielgroesse fuer Parameter-Sweeps.

        ### ANFORDERUNGEN AN DAS SKRIPT:
        1. CLI-SCHNITTSTELLE: Das Skript MUSS folgende Argumente per argparse akzeptieren:
           --params: Pfad zu einer JSON-Datei mit den Eingangsparametern.
           --output: Pfad, an den das Skript eine JSON-Datei mit den Ergebnissen schreiben MUSS.
           --plot: Pfad, an den das Skript ein PNG-Diagramm der Lösung speichern MUSS.
           
        2. NUMERISCHE METHODE:
           Wähle ein geeignetes numerisches Verfahren (z.B. scipy.integrate.solve_bvp für 1D BVPs, Finite Differenzen (FDM) für 2D, oder scipy.integrate.solve_ivp/odeint bei Anfangswertproblemen).
           Falls es sich um ein 2D-Problem handelt (z.B. Wärmeverteilung oder Potentialfeld), löse es auf einem 20x20 Gitter.
           Falls es sich um ein 1D-Problem handelt, verwende mindestens 20 Gitterpunkte.
           Berechne zusätzlich eine ungestörte Referenzlösung (z.B. indem du den Einfluss von slip_length, Wärmewiderstand oder Leitfähigkeit auf 0 setzt), um den Performance-Gewinn (Reibungsreduktion, Isolationswirkung etc.) zu bestimmen.
           
        3. PLOT-STYLING:
           Erzeuge ein professionelles Diagramm mit Matplotlib:
           - Hintergrund dunkel/transparent oder passend zum Dark Mode.
           - Verwende neon-farbene Kurven (z.B. Cyan für das bionische Modell, Magenta für die Referenz).
           - Beschrifte Achsen und Legenden vollständig und füge ein Raster (Grid) hinzu.
           
        4. ERGEBNIS-DATENSTRUKTUR (JSON an --output):
           Die Ausgabedatei MUSS exakt diese JSON-Struktur haben:
           {{
             "success": true,
             "performance_gain_pct": float (Prozentuale Verbesserung gegenüber der Referenz, z.B. 45.2),
             "relative_error": float (Numerischer Fehlerschätzer der bionischen Lösung als Bruchteil, z.B. Residuum, Energie-/Massenbilanz-Fehler oder Gitterkonvergenz N vs N/2 - NICHT der Unterschied zwischen bionischer und Referenzlösung; Varianten mit relative_error > {MAX_SWEEP_RELATIVE_ERROR} werden verworfen),
             "sample_points": List (Bei 1D: Liste von Floats. Bei 2D: Liste von [x, y] Koordinatenpaaren),
             "solution_primary": List[float] (Numerisch gelöste Feldwerte an den sample_points),
             "solution_reference": List[float] (Referenz-Feldwerte an den sample_points),
             "primary_metric_value": float (Berechnete physikalische Hauptkennzahl für das bionische System, z.B. Wandschubspannung oder Wärmestrom),
             "reference_metric_value": float (Hauptkennzahl für das Referenzsystem)
           }}
           Setze "success": false nur, wenn die Rechnung fehlgeschlagen ist; das zählt als Fehlschlag und löst eine Korrektur aus.
           Zusätzliche numerische Felder (z.B. Spitzenwerte, die in den hard_constraints vorkommen) sind erlaubt und erwünscht;
           benenne sie so, dass Constraints der Form `<feldname> <= <zahl>` direkt dagegen geprüft werden können.
           
        5. DATEI-KODIERUNG:
            Alle Lese- und Schreiboperationen auf Dateien (wie das Einlesen von --params und Schreiben von --output) MÜSSEN explizit mit `encoding="utf-8"` geöffnet werden (z.B. open(..., 'w', encoding='utf-8') oder open(..., 'r', encoding='utf-8')). Das ist zwingend erforderlich, um Codierungsfehler auf Windows-Systemen zu vermeiden.
            
        6. SANDBOX:
            {SANDBOX_RULES_TEXT}
            Das Skript läuft in einem temporären Arbeitsverzeichnis und darf nur die über --params/--output/--plot übergebenen Dateien lesen/schreiben.
            Verwende ausschließlich numerische Bibliotheken (numpy, scipy, sympy, matplotlib, math, json, argparse, ...).

         Schreibe sauberen, robusten Python 3.13 Code. Fange mögliche Division-by-Zero Fehler ab.
         """

        correction_iteration = 0
        max_iterations = 3
        current_prompt = prompt
        
        while correction_iteration < max_iterations:
            print(f"[*] Generating script iteration {correction_iteration + 1}...")
            response = self.client.models.generate_content(
                model=get_model_name("script_generator"),
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=get_thinking_config("script_generator"),
                    response_mime_type="application/json",
                    response_schema=GeneratedScriptResponse,
                )
            )
            
            parsed: GeneratedScriptResponse = response.parsed
            code = parsed.code
            
            # Markdown fences entfernen falls das LLM sie trotz Schema eingefügt hat
            if code.startswith("```python"):
                code = code.replace("```python", "", 1)
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()

            # Skript auf Festplatte schreiben
            with open(script_path, "w", encoding="utf-8") as sf:
                sf.write(code)

            failure_kind = "crashed"
            violations = check_generated_code(code)
            if violations:
                failure_kind = "rejected"
                error_msg = format_rejection(violations)
                print(f"[-] Script rejected by sandbox pre-check on iteration {correction_iteration + 1}!")
                print(f"[-] {error_msg}")
            else:
                print(f"[*] Executing generated script at: {script_path}")
                results_data, error_msg, failure_kind = self._execute(
                    script_path, params_json_path, output_json_path, plot_png_path
                )
                if results_data is not None:
                    print(f"[+] Script execution succeeded on iteration {correction_iteration + 1}!")
                    results_data["script_path"] = script_path
                    results_data["params_json_path"] = params_json_path
                    results_data["execution_mode"] = "generated_python_sandboxed_subprocess"
                    results_data["plot_png_path"] = plot_png_path
                    return results_data
                print(f"[-] Script failed on iteration {correction_iteration + 1} ({failure_kind})!")
                print(f"[-] Error: {error_msg}")

            # Vorbereitung der Korrekturschleife
            correction_iteration += 1
            if correction_iteration < max_iterations:
                print(f"[*] Initiating self-correction. Sending error details back to Gemini...")
                headline = {
                    "rejected": "Das zuvor generierte Python-Skript wurde von der Sandbox-Vorprüfung abgelehnt und nicht ausgeführt!",
                    "invalid_output": "Das zuvor generierte Python-Skript lief durch, aber sein Ergebnis-JSON erfüllt den Vertrag nicht (z.B. \"success\": false oder fehlende/nicht-endliche Felder)!",
                }.get(failure_kind, "Das zuvor generierte Python-Skript ist beim Ausführen abgestürzt!")
                current_prompt = f"""
                {headline}
                
                ### Zuvor generierter Code:
                ```python
                {code}
                ```
                
                ### Fehlermeldung (Traceback / Stderr / Prüfergebnis):
                ```text
                {error_msg}
                ```
                
                ### Aufgabe:
                Analysiere den Fehler, korrigiere den Code und liefere ein repariertes, vollständig lauffähiges Skript zurück.
                Halte dich strikt an die CLI-Parameter (--params, --output, --plot), die Ausgabestruktur und die Sandbox-Regeln:
                {SANDBOX_RULES_TEXT}
                """
        
        # Falls alle Iterationen fehlgeschlagen sind
        raise RuntimeError(f"Dynamic script solver execution failed after {max_iterations} attempts. Last error: {error_msg}")


    @staticmethod
    def _execute(script_path: str, params_json_path: str, output_json_path: str, plot_png_path: str):
        """
        Run the solver in the sandbox. Returns (results, error_msg, failure_kind); results
        is None on failure. Exit code 0 alone is not enough: the output JSON must exist,
        report "success": true and contain the contract fields with finite values.
        """
        if os.path.exists(output_json_path):
            os.remove(output_json_path)
        with sandbox_workdir("vectornaut_solver_") as workdir:
            copy_if_exists(script_path, os.path.join(workdir, "solver.py"))
            copy_if_exists(params_json_path, os.path.join(workdir, "params.json"))
            run = run_generated_script(
                ["solver.py", "--params", "params.json", "--output", "results.json", "--plot", "plot.png"],
                timeout=script_timeout_seconds(),
                workdir=workdir,
            )
            copy_if_exists(os.path.join(workdir, "results.json"), output_json_path)
            copy_if_exists(os.path.join(workdir, "plot.png"), plot_png_path)
        if run.timed_out:
            return None, f"Generated script timed out after {run.duration_s:.0f} seconds (process group killed).", "crashed"
        if run.returncode != 0:
            return None, run.error_text() or f"Exit code {run.returncode}", "crashed"
        try:
            with open(output_json_path, "r", encoding="utf-8") as rf:
                results_data = json.load(rf)
        except Exception as read_err:
            return None, f"Failed to read results JSON: {read_err}", "invalid_output"
        problems = solver_output_problems(results_data)
        if problems:
            detail = "; ".join(problems)
            if isinstance(results_data, dict) and results_data.get("error"):
                detail += f"; script error: {results_data.get('error')}"
            return None, f"Exit code 0, but the output violates the result contract: {detail}", "invalid_output"
        return results_data, "", ""
