# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from vectornaut.storage import history_dir, index_history_run, reports_dir


def archive_run_data(response_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    try:
        os.makedirs(history_dir(), exist_ok=True)
        os.makedirs(reports_dir(), exist_ok=True)

        design_name = response_data.get("miner", {}).get("design_name") or "unknown_design"
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_content = generate_markdown_report_content(response_data, timestamp)
        response_data["report_md"] = md_content

        md_filename = f"report_{timestamp}_{slug}.md"
        md_path = os.path.join(reports_dir(), md_filename)
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(md_content)

        json_filename = f"run_{timestamp}_{slug}.json"
        json_path = os.path.join(history_dir(), json_filename)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(response_data, jf, indent=4, ensure_ascii=False)

        run_id = os.path.basename(json_path).replace(".json", "")
        index_history_run(run_id, json_path, md_path, response_data)
        print(f"[*] Archived run data to {json_path} and {md_path}")
        return {"history_path": json_path, "report_path": md_path}
    except Exception as archive_err:
        print(f"[*] Archiving failed: {archive_err}")
        return None

def generate_markdown_report_content(data: dict, timestamp_str: str) -> str:
    miner = data.get("miner", {})
    auditor = data.get("auditor", {})
    simulator = data.get("simulator", {})
    synthesis = data.get("synthesis", {})
    validation = data.get("validation", {})
    
    from datetime import datetime
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        readable_time = timestamp_str

    parameters = miner.get("parameters", [])
    audited_params = auditor.get("audited_parameters", [])
    dim_numbers = auditor.get("dimensionless_numbers", [])
    
    md = []
    md.append(f"# Bionisches Design-Protokoll: {miner.get('design_name', 'Unbenanntes Design')}")
    md.append(f"*Erstellt am: {readable_time} (Vectornaut Engine)*\n")
    
    # Executive Summary & Pros/Cons at the top
    if synthesis.get("executive_summary"):
        md.append("## Executive Summary")
        md.append(synthesis.get("executive_summary"))
        md.append("")
        
    if synthesis.get("pros_and_cons"):
        md.append("## Vor- & Nachteile (Gegenüberstellung)")
        md.append(synthesis.get("pros_and_cons"))
        md.append("")

    if validation:
        score = validation.get("score")
        score_text = f"{score * 100:.0f}%" if isinstance(score, (int, float)) else "N/A"
        action_labels = {
            "accept": "Ergebnis übernehmen",
            "inspect": "Warnungen prüfen",
            "rerun_solver": "Mit alternativem Solver vergleichen",
            "remine": "Neues Konzept suchen",
        }
        action = validation.get("recommended_action", "inspect")
        md.append("## Prüfentscheidung des Validators")
        md.append(f"- **Status:** `{str(validation.get('status', 'unknown')).upper()}`")
        md.append(f"- **Verlässlichkeit:** `{str(validation.get('reliability', 'unknown')).upper()}`")
        md.append(f"- **Prüfscore:** `{score_text}`")
        md.append(f"- **Empfohlene Aktion:** **{action_labels.get(action, action)}**")
        md.append("")
        md.append("Der Prüfscore ist der Anteil bestandener technischer Checks. Er ist keine Wahrscheinlichkeit, dass das Ergebnis real korrekt ist.")
        warnings = validation.get("warnings", [])
        if warnings:
            md.append("\n### Wichtigste Validator-Hinweise")
            for warning in warnings[:5]:
                md.append(f"- {warning}")
        md.append("")
    
    md.append("## 1. Konzept & Bionische Inspiration")
    md.append(f"- **Natürliches Vorbild:** {miner.get('inspiration_source', 'N/A')}")
    md.append(f"- **Physikalisches System (Domäne):** {miner.get('domain', 'N/A')}")
    md.append(f"- **Bionischer Mechanismus:**\n  > {miner.get('physical_mechanism', 'N/A')}\n")
    
    # Embed the SVG Construction Schematic
    svg_schematic = miner.get("svg_schematic")
    if svg_schematic:
        md.append("### Schematische Konstruktionszeichnung")
        md.append(f'<div class="report-svg-container" style="background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 15px 0; max-width: 600px;">{svg_schematic}</div>\n')
    
    md.append("## 2. Mathematische Formulierung")
    is_2d = len(miner.get("independent_variables", [])) == 2
    if is_2d:
        md.append("Das System beschreibt das Verhalten über eine zweidimensionale (2D) partielle Differenzialgleichung (PDE):")
    else:
        md.append("Das System beschreibt das Verhalten über ein eindimensionales (1D) Randwertproblem (Boundary Value Problem):")
    md.append(f"- **Differenzialgleichung (PDE/ODE):** `{miner.get('governing_equation', 'N/A')}`")
    md.append("- **Randbedingungen (Boundary Conditions):**")
    for bc in miner.get("boundary_conditions", []):
        md.append(f"  * `{bc}`")
    ind_vars_str = ", ".join(miner.get("independent_variables", ["x"]))
    md.append(f"- **Variablen:** Unabhängig: `{ind_vars_str}`, Abhängig: `{miner.get('dependent_variables', ['y'])[0]}`\n")
    
    md.append("## 3. Parameter-Audit & Kennzahlen")
    md.append("| Parameter | Vorgeschlagener Wert (Miner) | Geprüfter Wert (Auditor) | Zulässige Grenzen | Begründung |")
    md.append("| --- | --- | --- | --- | --- |")
    
    audited_dict = {p["name"]: p["value"] for p in audited_params}
    for p in parameters:
        p_name = p.get("name", "")
        min_v = p.get("min_bound", 0.0)
        max_v = p.get("max_bound", 0.0)
        just = p.get("justification", "")
        mined_val = p.get("value", 0.0)
        audited_val = audited_dict.get(p_name, mined_val)
        
        md.append(f"| `{p_name}` | {mined_val} | **{audited_val}** | `[{min_v}, {max_v}]` | {just} |")
    
    md.append("\n### Berechnete dimensionslose Kennzahlen")
    for dim in dim_numbers:
        md.append(f"- **{dim.get('name', 'N/A')}:** {dim.get('value', 0.0)}")
    
    md.append(f"\n- **Abgeleiteter Simulationskoeffizient:** `{auditor.get('simulation_coefficient', 0.0)}` (genutzt in den Gleichungen)")
    md.append(f"- **Freigegebener Solver:** `{auditor.get('solver_method', 'N/A').upper()}`\n")
    
    md.append("## 4. Audit-Bewertung (Auditor Notes)")
    md.append(f"> {auditor.get('audit_notes', 'Keine Audit-Notizen vorhanden.')}\n")
    
    md.append("## 5. Simulationsergebnisse")
    md.append(f"- **Lösungsmethode:** `{simulator.get('solver_method', 'N/A').upper()}`")
    
    ui_meta = auditor.get("ui_metadata", {})
    gain_label = ui_meta.get("performance_gain", {}).get("label", "Effizienzsteigerung")
    gain_val = simulator.get("performance_gain_pct", 0.0)
    md.append(f"- **{gain_label}:** **{gain_val:.2f}%**")
    
    prim_label = ui_meta.get("primary_metric", {}).get("label", "Primäre Metrik")
    prim_val = simulator.get("primary_metric_value", 0.0)
    ref_label = ui_meta.get("reference_metric", {}).get("label", "Referenzmetrik")
    ref_val = simulator.get("reference_metric_value", 0.0)
    
    md.append(f"- **{prim_label} (Bionisch):** `{prim_val:.4f}`")
    md.append(f"- **{ref_label} (Referenz):** `{ref_val:.4f}`")
    
    if simulator.get("solver_method") == "pinn":
        md.append(f"- **PINN Trainingsepochen:** {simulator.get('epochs_trained', 0)}")
        md.append(f"- **PINN End-Loss (MSE):** `{simulator.get('final_loss', 0.0):.4e}`")
    elif simulator.get("solver_method") == "dynamic_script":
        md.append("\n### KI-generierter Python-Solver")
        md.append("- **Ausführung:** generiertes Python-Skript in separatem Subprocess")
        md.append(f"- **Solver-Skript:** `{os.path.basename(simulator.get('script_path') or 'N/A')}`")
        md.append(f"- **Parameterdatei:** `{os.path.basename(simulator.get('params_json_path') or 'N/A')}`")
        md.append(f"- **Testskript:** `{os.path.basename(simulator.get('test_script_path') or 'N/A')}`")
        md.append(f"- **Testresultat:** `{os.path.basename(simulator.get('test_output_path') or 'N/A')}`")
        md.append(f"- **Automatische Tests bestanden:** `{simulator.get('validation_passed')}`")
        
    custom_plot = simulator.get("custom_plot_url")
    if custom_plot:
        md.append("\n### Simulationsdiagramm (Feldverteilung)")
        # Relative image URL for clean rendering in browser and pdf exports
        md.append(f'![Simulations-Plot]({custom_plot})')
    
    md.append("\n### Berechnete Stützpunkte und Feldwerte")
    loc_header = "Ort (x, y)" if is_2d else "Ort (x)"
    md.append(f"| {loc_header} | Bionischer Wert (Primary) | Referenzwert (Reference) | Abweichung |")
    md.append("| --- | --- | --- | --- |")
    
    pts = simulator.get("sample_points", [])
    sol_p = simulator.get("solution_primary", [])
    sol_r = simulator.get("solution_reference", [])
    
    for i in range(len(pts)):
        p_val = sol_p[i] if i < len(sol_p) else 0.0
        r_val = sol_r[i] if i < len(sol_r) else 0.0
        diff = p_val - r_val
        if is_2d:
            try:
                coord_str = f"({pts[i][0]:.3f}, {pts[i][1]:.3f})"
            except Exception:
                coord_str = str(pts[i])
        else:
            try:
                coord_str = f"{pts[i]:.4f}"
            except Exception:
                coord_str = str(pts[i])
        md.append(f"| {coord_str} | {p_val:.4f} | {r_val:.4f} | {diff:.4f} |")
        
    if simulator.get("validation_report"):
        md.append("\n## 6. Automatische Validierung (AI-Generated Tests)")
        md.append(simulator.get("validation_report"))

    # Append Section 7: Optimization History if present
    opt_hist = data.get("optimization_history", [])
    if opt_hist:
        md.append("\n## 7. Autonome Optimierungshistorie (Closed-Loop)")
        md.append("Das System hat die Parameter in mehreren Simulations- und Validierungsschleifen autonom angepasst:\n")
        md.append("| Runde | Parameter-Set | Koeffizient | Effizienz (Gain) | Validierung | Feedback des Optimierers |")
        md.append("| --- | --- | --- | --- | --- | --- |")
        for run in opt_hist:
            r_num = run.get("round", 1)
            params_str = ", ".join([f"`{k}`: {v}" for k, v in run.get("parameters", {}).items()])
            coeff_val = run.get("simulation_coefficient", 0.0)
            sim_res = run.get("simulator", {})
            gain = sim_res.get("performance_gain_pct", 0.0)
            val_status = "✅ PASS" if sim_res.get("validation_passed") else "❌ FAIL" if sim_res.get("validation_passed") is not None else "N/A"
            if run.get("validation", {}).get("status"):
                val_status = str(run.get("validation", {}).get("status")).upper()
            reasoning = run.get("optimizer_reasoning", "Konvergenz erreicht oder Limit erreicht.").replace("\n", " ").strip()
            md.append(f"| {r_num} | {params_str} | `{coeff_val:.6f}` | **{gain:.2f}%** | {val_status} | {reasoning} |")

            fallbacks = run.get("solver_fallbacks", [])
            if fallbacks:
                fallback_text = "; ".join(
                    f"{fb.get('solver_method', 'unknown').upper()}: {fb.get('status', 'failed')} (score={fb.get('score', 'N/A')})"
                    for fb in fallbacks
                )
                md.append(f"\nSolver-Fallbacks Runde {r_num}: {fallback_text}\n")

    # Append Section 8: Commercial & Engineering Synthesis
    synthesis = data.get("synthesis", {})
    if synthesis:
        md.append("\n## 8. Kommerzielle & Praktische Synthese")
        md.append("Hier ist die ingenieurwissenschaftliche und wirtschaftliche Bewertung für die Umsetzung dieses Entwurfs:")
        md.append(f"\n### Mechanische Belastbarkeit & Sicherheitsgrenzen\n{synthesis.get('mechanical_limits')}")
        md.append(f"\n### Empfohlene Fertigungsmethoden & Skalierung\n{synthesis.get('manufacturing_methods')}")
        md.append(f"\n### Kostenschätzung (Prototyping & Produktion)\n{synthesis.get('cost_estimation')}")
        md.append(f"\n### Vorgeschlagene Validierungsexperimente\n{synthesis.get('validation_experiments')}")
        md.append(f"\n### Mögliche Partner & Industriebranchen\n{synthesis.get('industry_partners')}")

    # Append Section 9: Failed Concepts (Re-Mining History) if present
    failed_c = data.get("failed_concepts", [])
    if failed_c:
        md.append("\n## 9. Verlauf gescheiterter Konzepte (Re-Mining)")
        md.append("Die folgenden bionischen Ansätze wurden während der Pipeline evaluiert, aber aufgrund mangelnder Stabilität oder Plausibilität verworfen:")
        for idx, fc in enumerate(failed_c, 1):
            md.append(f"\n* **Ansatz {idx}: {fc.get('design_name')}** ({fc.get('inspiration_source')})")
            md.append(f"  * *Grund für das Scheitern:* {fc.get('reason')}")
        
    return "\n".join(md)
