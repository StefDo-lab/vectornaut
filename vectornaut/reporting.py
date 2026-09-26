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

        is_rejected = response_data.get("status") == "rejected"
        design_name = (response_data.get("miner") or {}).get("design_name") or ("rejected_request" if is_rejected else "unknown_design")
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_rejected:
            md_content = generate_rejection_report_content(response_data, timestamp)
        else:
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

# Coefficient names the equations may use for the auditor's simulation_coefficient
# (see solver_dispatcher._merged_params; aliases only when not audited explicitly).
_COEFFICIENT_ALIASES = ("slippage_coefficient", "lambda", "slip_length")

# Longest solution table in the report; larger (2D) grids are subsampled.
MAX_SOLUTION_TABLE_ROWS = 25


def _fmt_num(value: Any, spec: str = ".6g") -> str:
    """Significant-figure formatting for report numbers; non-numbers are returned as text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return format(float(value), spec)


def _with_unit(value: Any, unit: Optional[str]) -> str:
    text = _fmt_num(value)
    return f"{text} {unit}" if unit else text


def gain_available(simulator: Dict[str, Any]) -> bool:
    """False if the solver reported that no baseline exists (performance_gain_pct means n/a)."""
    return (simulator or {}).get("gain_basis") != "none"


# Same default as the validator's physics_performance_gain_sanity check.
DEFAULT_MAX_ABS_GAIN_PCT = 500.0
GAIN_SANITY_CHECK = "physics_performance_gain_sanity"


def gain_implausible(simulator: Dict[str, Any], validation: Optional[Dict[str, Any]] = None) -> bool:
    """
    True if the gain exists but failed the validator's gain-sanity check. The check result in
    `validation` (dict or ValidationResult) decides; without one, |gain| > 500 % (the
    validator's default limit) or a non-finite gain counts as implausible.
    """
    simulator = simulator or {}
    if not gain_available(simulator):
        return False
    if validation is not None and not isinstance(validation, dict):
        validation = validation.model_dump() if hasattr(validation, "model_dump") else dict(getattr(validation, "__dict__", {}))
    for check in (validation or {}).get("checks", []) or []:
        check = check if isinstance(check, dict) else (check.model_dump() if hasattr(check, "model_dump") else {})
        if check.get("name") == GAIN_SANITY_CHECK:
            return not check.get("passed", True)
    try:
        gain = float(simulator.get("performance_gain_pct"))
    except (TypeError, ValueError):
        return True
    return not (abs(gain) <= DEFAULT_MAX_ABS_GAIN_PCT)


def format_gain(simulator: Dict[str, Any], validation: Optional[Dict[str, Any]] = None) -> str:
    if not gain_available(simulator):
        return "n/a"
    text = f"{float((simulator or {}).get('performance_gain_pct', 0.0) or 0.0):.2f}%"
    if gain_implausible(simulator, validation):
        return f"implausibel ({text})"
    return text


def _coefficient_used(miner: Dict[str, Any], auditor: Dict[str, Any]) -> bool:
    """True if the governing equation or a BC refers to simulation_coefficient or one of its aliases."""
    audited = {p.get("name") for p in auditor.get("audited_parameters", []) or [] if isinstance(p, dict)}
    names = ["simulation_coefficient"] + [a for a in _COEFFICIENT_ALIASES if a not in audited]
    text = " ".join([str(miner.get("governing_equation") or "")] + [str(bc) for bc in miner.get("boundary_conditions", []) or []])
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) for name in names)


def _spread(count: int, limit: int) -> list:
    """All indices 0..count-1, or `limit` evenly spread ones including the first and the last."""
    if count <= limit:
        return list(range(count))
    step = (count - 1) / (limit - 1)
    return sorted({int(round(i * step)) for i in range(limit)})


def _table_indices(points: list, limit: int = MAX_SOLUTION_TABLE_ROWS) -> list:
    """
    Row indices of the solution table: all rows up to `limit`, otherwise an evenly spread
    subset. For 2D grids the subset is a coarser grid (e.g. 5 x 5 of 20 x 20 nodes).
    """
    count = len(points)
    if count <= limit:
        return list(range(count))
    try:
        xs = sorted({float(p[0]) for p in points})
        ys = sorted({float(p[1]) for p in points})
    except (TypeError, ValueError, IndexError):
        return _spread(count, limit)
    if len(xs) * len(ys) != count:
        return _spread(count, limit)
    per_axis = max(2, int(limit ** 0.5))
    keep_x = {xs[i] for i in _spread(len(xs), per_axis)}
    keep_y = {ys[i] for i in _spread(len(ys), per_axis)}
    return [i for i, p in enumerate(points) if float(p[0]) in keep_x and float(p[1]) in keep_y]


_REJECTION_STAGE_LABELS = {
    "miner": "Konzeptsuche (Miner)",
    "auditor": "Physik-Audit (Auditor)",
    "validator": "Deterministische Physikprüfung (Validator)",
}


def generate_rejection_report_content(data: dict, timestamp_str: str) -> str:
    """Report for a run that stopped because the request is physically impossible as stated."""
    rejection = data.get("rejection") or {}
    miner = data.get("miner") or {}
    auditor = data.get("auditor") or {}
    try:
        readable_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        readable_time = timestamp_str

    stage = rejection.get("stage", "unknown")
    md = [
        "# Anfrage abgelehnt: physikalisch nicht umsetzbar",
        f"*Erstellt am: {readable_time} (Vectornaut Engine)*\n",
        "## Begründung",
        f"> {rejection.get('reason') or 'Keine Begründung angegeben.'}\n",
        f"- **Abgelehnt durch:** {_REJECTION_STAGE_LABELS.get(stage, stage)}",
        f"- **Konzeptversuch:** {rejection.get('concept_attempt', 'N/A')}",
        f"- **Anfrage:** {data.get('query') or 'N/A'}",
        "",
        "Die Pipeline hat nach dieser Feststellung keine weiteren Konzepte gesucht und keine Ergebnisse als Lösung der Anfrage ausgegeben, "
        "weil kein Konzept eine physikalisch unmögliche Anforderung erfüllen kann.",
    ]
    if miner.get("design_name"):
        md.append("\n## Zuletzt betrachtetes Konzept (nicht als Lösung zu verstehen)")
        md.append(f"- **Konzept:** {miner.get('design_name')}")
        if miner.get("inspiration_source"):
            md.append(f"- **Vorbild:** {miner.get('inspiration_source')}")
        if miner.get("domain"):
            md.append(f"- **Domäne:** {miner.get('domain')}")
        if miner.get("physical_mechanism"):
            md.append(f"- **Mechanismus:**\n  > {miner.get('physical_mechanism')}")
    if auditor.get("audit_notes"):
        md.append("\n## Audit-Notizen")
        md.append(f"> {auditor.get('audit_notes')}")
    failed_c = data.get("failed_concepts") or []
    if failed_c:
        md.append("\n## Vorher verworfene Konzepte")
        for idx, fc in enumerate(failed_c, 1):
            md.append(f"* **Ansatz {idx}: {fc.get('design_name')}** ({fc.get('inspiration_source')}): {fc.get('reason')}")
    return "\n".join(md)


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
        
        md.append(f"| `{p_name}` | {_fmt_num(mined_val)} | **{_fmt_num(audited_val)}** | `[{_fmt_num(min_v)}, {_fmt_num(max_v)}]` | {just} |")
    
    md.append("\n### Berechnete dimensionslose Kennzahlen")
    for dim in dim_numbers:
        md.append(f"- **{dim.get('name', 'N/A')}:** {dim.get('value', 0.0)}")
    
    coefficient_usage = "genutzt in den Gleichungen" if _coefficient_used(miner, auditor) else "in den Gleichungen nicht verwendet"
    md.append(f"\n- **Abgeleiteter Simulationskoeffizient:** `{_fmt_num(auditor.get('simulation_coefficient', 0.0))}` ({coefficient_usage})")
    md.append(f"- **Freigegebener Solver:** `{auditor.get('solver_method', 'N/A').upper()}`\n")
    objective_contract = auditor.get("objective_metric") or simulator.get("objective_metric") or {}
    if objective_contract:
        md.append("### Bewertungsvertrag (Metric Contract)")
        md.append(f"- **Zielmetrik:** `{objective_contract.get('objective_name', 'Performance Gain')}`")
        md.append(f"- **Score-Feld:** `{objective_contract.get('score_field', 'performance_gain_pct')}`")
        md.append(f"- **Richtung:** `{objective_contract.get('direction', 'maximize')}`")
        md.append(f"- **Primary Metric:** `{objective_contract.get('primary_metric', 'Primary Metric')}`")
        md.append(f"- **Reference Metric:** `{objective_contract.get('reference_metric', 'Reference Metric')}`")
        md.append(f"- **Acceptance Threshold:** `{objective_contract.get('acceptance_threshold', 0.0)}`")
        constraints = objective_contract.get("hard_constraints") or []
        if constraints:
            md.append("- **Harte Nebenbedingungen:** " + "; ".join(f"`{c}`" for c in constraints))
        md.append("")
    
    md.append("## 4. Audit-Bewertung (Auditor Notes)")
    md.append(f"> {auditor.get('audit_notes', 'Keine Audit-Notizen vorhanden.')}\n")
    
    md.append("## 5. Simulationsergebnisse")
    md.append(f"- **Lösungsmethode:** `{simulator.get('solver_method', 'N/A').upper()}`")
    
    ui_meta = auditor.get("ui_metadata", {})
    gain_label = ui_meta.get("performance_gain", {}).get("label", "Effizienzsteigerung")
    metric_spec = simulator.get("metric_spec") or {}
    metric_unit = simulator.get("metric_unit") or metric_spec.get("unit")
    if gain_implausible(simulator, validation):
        md.append(
            f"- **{gain_label}:** **implausibel** (berechnet {_fmt_num(simulator.get('performance_gain_pct'))} %, "
            f"der Gewinn hat die Plausibilitätsprüfung des Validators (Standardgrenze |Gewinn| <= 500 %) nicht bestanden; "
            f"kein belastbares Ergebnis, Metrik und Vergleichsdesign prüfen)"
        )
    elif gain_available(simulator):
        md.append(f"- **{gain_label}:** **{format_gain(simulator)}**")
    else:
        md.append(f"- **{gain_label}:** **n/a** (kein Vergleichsdesign definiert, der Gewinn ist nicht berechenbar)")

    prim_label = ui_meta.get("primary_metric", {}).get("label", "Primäre Metrik")
    prim_val = simulator.get("primary_metric_value", 0.0)
    ref_label = ui_meta.get("reference_metric", {}).get("label", "Referenzmetrik")
    ref_val = simulator.get("reference_metric_value", 0.0)

    md.append(f"- **{prim_label} (Bionisch):** `{_with_unit(prim_val, metric_unit)}`")
    md.append(f"- **{ref_label} (Referenz):** `{_with_unit(ref_val, metric_unit)}`")
    if metric_spec:
        location = f" bei `{metric_spec.get('location')}`" if metric_spec.get("location") else ""
        scale = f", skaliert mit `{metric_spec.get('scale')}`" if metric_spec.get("scale") else ""
        metric_name = metric_spec.get("label") or prim_label
        transform = f", Gütemaß `{metric_spec.get('transform')}` (m = Metrik)" if metric_spec.get("transform") else ""
        md.append(f"- **Metrik-Definition:** {metric_name}: `{metric_spec.get('kind')}`{location}{scale}{transform}")
    baseline_val = simulator.get("baseline_metric_value")
    if baseline_val is not None and gain_available(simulator):
        basis = simulator.get("gain_basis")
        if basis == "baseline_parameters":
            baseline_name = auditor.get("baseline_description") or "konventionelles Vergleichsdesign"
            overrides = ", ".join(
                f"`{p.get('name')}` = {_fmt_num(p.get('value'))}"
                for p in auditor.get("baseline_parameters") or [] if isinstance(p, dict)
            )
            baseline_name = f"{baseline_name} ({overrides})" if overrides else baseline_name
        else:
            baseline_name = "gleiches Design ohne bionischen Effekt (Koeffizient = 0)"
        md.append(f"- **Vergleichsdesign (Baseline):** {baseline_name}: `{_with_unit(baseline_val, metric_unit)}`")
    if simulator.get("gain_note"):
        md.append(f"- **Hinweis zur Metrik:** {simulator.get('gain_note')}")
    if simulator.get("solver_note"):
        md.append(f"- **Hinweis zum Solver (Fallback):** {simulator.get('solver_note')}")
    
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
        sweep = simulator.get("parameter_sweep") or {}
        objective = sweep.get("objective") or simulator.get("objective_metric") or auditor.get("objective_metric") or {}
        if sweep:
            best = sweep.get("best") or {}
            md.append("\n### Dynamic-Script Parameter-Sweep")
            md.append(f"- **Zielmetrik:** `{objective.get('objective_name', objective.get('name', 'Performance Gain'))}` ({objective.get('direction', 'maximize')})")
            md.append(f"- **Getestete Varianten:** {sweep.get('valid_candidate_count', 0)} verwertbar von {sweep.get('candidate_count', 0)}")
            md.append(f"- **Akzeptierte Varianten:** {sweep.get('accepted_candidate_count', 0)}")
            if best:
                md.append(f"- **Bestes Ergebnis:** `{best.get('label', 'N/A')}` mit `{best.get('performance_gain_pct', 'N/A')}%` Gain")
                changed = best.get("changed_parameter")
                if changed:
                    md.append(f"- **Bester Parameter:** `{changed} = {best.get('changed_value')}`")
        
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
    
    row_indices = _table_indices(pts) if is_2d else _spread(len(pts), MAX_SOLUTION_TABLE_ROWS)
    for i in row_indices:
        p_val = sol_p[i] if i < len(sol_p) else 0.0
        r_val = sol_r[i] if i < len(sol_r) else 0.0
        diff = p_val - r_val
        if is_2d:
            try:
                coord_str = f"({_fmt_num(pts[i][0], '.4g')}, {_fmt_num(pts[i][1], '.4g')})"
            except Exception:
                coord_str = str(pts[i])
        else:
            try:
                coord_str = _fmt_num(pts[i], '.4g')
            except Exception:
                coord_str = str(pts[i])
        md.append(f"| {coord_str} | {_fmt_num(p_val)} | {_fmt_num(r_val)} | {_fmt_num(diff, '.3g')} |")
    if len(row_indices) < len(pts):
        md.append(f"\n*{len(row_indices)} von {len(pts)} Stützpunkten gezeigt (gleichmäßig verteilte Auswahl); alle Werte stehen im JSON-Export.*")
        
    if simulator.get("validation_report"):
        md.append("\n## 6. Automatische Validierung (AI-Generated Tests)")
        md.append(simulator.get("validation_report"))

    # Append Section 7: Optimization History if present
    opt_hist = data.get("optimization_history", [])
    if opt_hist:
        md.append("\n## 7. Autonome Optimierungshistorie (Closed-Loop)")
        md.append("Das System hat die Parameter in mehreren Simulations- und Validierungsschleifen autonom angepasst:\n")
        md.append("| Runde | Parameter-Set | Koeffizient | Metrik | Effizienz (Gain) | Validierung | Feedback des Optimierers |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for run in opt_hist:
            r_num = run.get("round", 1)
            params_str = ", ".join([f"`{k}`: {_fmt_num(v)}" for k, v in run.get("parameters", {}).items()])
            coeff_val = run.get("simulation_coefficient", 0.0)
            sim_res = run.get("simulator", {})
            metric_str = _with_unit(sim_res.get("primary_metric_value", 0.0), sim_res.get("metric_unit"))
            val_status = "✅ PASS" if sim_res.get("validation_passed") else "❌ FAIL" if sim_res.get("validation_passed") is not None else "N/A"
            if run.get("validation", {}).get("status"):
                val_status = str(run.get("validation", {}).get("status")).upper()
            reasoning = run.get("optimizer_reasoning", "Konvergenz erreicht oder Limit erreicht.").replace("\n", " ").strip()
            md.append(f"| {r_num} | {params_str} | `{_fmt_num(coeff_val)}` | {metric_str} | **{format_gain(sim_res, run.get('validation') or None)}** | {val_status} | {reasoning} |")

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
