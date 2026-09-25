# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .config import get_client, get_model_name, get_thinking_config, MinerOutput

class ParameterAdjustment(BaseModel):
    name: str = Field(description="Name des Parameters, der angepasst werden soll")
    value: float = Field(description="Der neue vorgeschlagene numerische Wert für diesen Parameter. MUSS streng innerhalb der min/max Schranken liegen.")

class OptimizerDecision(BaseModel):
    continue_optimization: bool = Field(description="True, wenn eine weitere Simulationsrunde sinnvoll ist, um den Entwurf weiter zu verbessern")
    reasoning: str = Field(description="Kurze Erklärung der physikalischen Begründung für diese Entscheidung und Anpassungen (auf Deutsch, informelles 'du')")
    adjustments: List[ParameterAdjustment] = Field(description="Liste der vorgeschlagenen Parameter-Anpassungen. Kann leer sein, wenn continue_optimization=False ist.", default_factory=list)
    concept_failed: bool = Field(default=False, description="True NUR, wenn das Konzept physikalisch oder strukturell versagt (z. B. Kollaps, Materialversagen, nicht behebbare Instabilität) und verworfen werden muss. Die bloße Feststellung, dass KEINE Instabilität vorliegt, ist kein Versagen.")

def _metric_summary(sim: Dict[str, Any]) -> str:
    """Prompt lines with the round's metric (name, value, unit, baseline) and gain."""
    spec = sim.get("metric_spec") or {}
    unit = sim.get("metric_unit") or spec.get("unit") or ""
    unit_text = f" {unit}" if unit else ""
    name = spec.get("label") or " ".join(str(spec[k]) for k in ("kind", "location") if spec.get(k)) or "Primärmetrik"
    lines = f"  * Metrik ({name}): {float(sim.get('primary_metric_value', 0.0) or 0.0):.6g}{unit_text}\n"
    if sim.get("baseline_metric_value") is not None:
        lines += f"  * Metrik des Vergleichsdesigns (Baseline): {float(sim['baseline_metric_value']):.6g}{unit_text}\n"
    if sim.get("gain_basis") == "none":
        lines += "  * Performance Gain: n/a (kein Vergleichsdesign definiert, der Gain ist nicht berechenbar; bewerte die Metrik selbst)\n"
    else:
        lines += f"  * Performance Gain: {float(sim.get('performance_gain_pct', 0.0) or 0.0):.2f}%\n"
    return lines


class Optimizer:
    def __init__(self, client=None):
        self.client = client

    def optimize(self, miner_output: MinerOutput, history: List[Dict[str, Any]]) -> OptimizerDecision:
        """
        Analysiert die bisherigen Simulationsläufe und entscheidet autonom über Parameteranpassungen.
        """
        # Aktueller Lauf (letzter Eintrag in der Historie)
        from google.genai import types

        latest_run = history[-1]
        round_num = latest_run["round"]
        sim_data = latest_run["simulator"]
        
        # Bisherige Runden zusammenfassen
        history_summary = ""
        for run in history:
            history_summary += f"- Runde {run['round']}:\n"
            history_summary += f"  * Parameter: {run['parameters']}\n"
            history_summary += _metric_summary(run['simulator'])
            history_summary += f"  * Relative Error: {run['simulator'].get('relative_error', 0.0):.4e}\n"
            if run['simulator'].get('validation_passed') is not None:
                history_summary += f"  * Validierung bestanden: {run['simulator'].get('validation_passed')}\n"
        
        # Parameter mit Schranken auflisten
        param_bounds = ""
        for p in miner_output.parameters:
            param_bounds += f"- Parameter: `{p.name}`\n"
            param_bounds += f"  * Aktueller Wert: {latest_run['parameters'].get(p.name)}\n"
            param_bounds += f"  * Erlaubte Schranken: [{p.min_bound}, {p.max_bound}]\n"
            param_bounds += f"  * Bionische Begründung: {p.justification}\n"

        prompt = f"""
        Du bist der bionische Materialoptimierer von Vectornaut. Deine Aufgabe ist es, die Simulationsergebnisse eines bionischen Materialentwurfs zu analysieren und zu entscheiden, ob eine weitere Optimierung der physikalischen Parameter sinnvoll ist.

        ### Bionisches Konzept
        Name: {miner_output.design_name}
        Natürliches Vorbild: {miner_output.inspiration_source}
        Domäne: {miner_output.domain}
        Physikalischer Mechanismus: {miner_output.physical_mechanism}
        Governing Equation: {miner_output.governing_equation}
        Randbedingungen: {miner_output.boundary_conditions}

        ### Aktuelle Parameter und Schranken
        {param_bounds}

        ### Historie der bisherigen Simulationsläufe
        {history_summary}

        ### DEINE AUFGABE:
        Analysiere die Daten physikalisch. Triff eine rationale Entscheidung, ob wir die Parameter weiter anpassen sollten, um eine höhere Effizienz (Performance Gain) zu erzielen, ohne physikalische Grenzwerte zu verletzen oder numerische Instabilitäten (z. B. relative Fehler > 1.0) hervorzurufen.
        
        Achte besonders auf:
        1. **Verbesserungspotenzial**: Haben wir das theoretische Limit bereits erreicht? Konvergiert die Verbesserung (z. B. kaum noch Änderung zwischen den Runden)?
        2. **Validierungstests**: Schlagen physikalische Invariantentests in der letzten Runde fehl? Wenn ja, passe die Parameter so an, dass sie wieder in den sicheren Bereich kommen, oder beende die Optimierung, falls keine Verbesserung möglich ist.
        3. **Grenzwerte**: Alle vorgeschlagenen Anpassungen MÜSSEN streng innerhalb der erlaubten Schranken liegen! Schlage niemals Werte außerhalb von [min_bound, max_bound] vor.

        Setze `continue_optimization` auf `True`, wenn wir eine weitere Runde drehen sollten, und liefere die Anpassungen in `adjustments`. Liefere andernfalls `False` und erkläre, warum das Design optimal ist oder nicht weiter verbessert werden kann.
        Setze `concept_failed` nur dann auf `True`, wenn das Konzept grundsätzlich versagt (z. B. struktureller Kollaps oder Materialversagen) und durch ein neues Konzept ersetzt werden muss. In allen anderen Fällen bleibt es `False`.
        """

        client = self.client or get_client()
        response = client.models.generate_content(
            model=get_model_name("optimizer"),
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=get_thinking_config("optimizer"),
                response_mime_type="application/json",
                response_schema=OptimizerDecision,
            )
        )
        return response.parsed

    def mock_optimize(self, miner_output: MinerOutput, history: List[Dict[str, Any]]) -> OptimizerDecision:
        """
        Mock-Implementierung für den Offline-Modus und Unit-Tests.
        """
        latest_run = history[-1]
        round_num = latest_run["round"]
        current_params = latest_run["parameters"]
        
        # Wenn das Design "fail" oder "instabil" im Namen hat, simulieren wir in Runde 1 einen mechanischen Kollaps
        design_name_lower = (miner_output.design_name or "").lower()
        if ("fail" in design_name_lower or "instabil" in design_name_lower) and "alternative" not in design_name_lower:
            return OptimizerDecision(
                continue_optimization=False,
                reasoning="Kritischer struktureller Kollaps: Die Geometrie ist unter den mechanischen Lasten eingeknickt. Konzept ist instabil.",
                adjustments=[],
                concept_failed=True
            )

        # Wenn wir Runde 1 abgeschlossen haben, schlagen wir eine Anpassung vor
        if round_num == 1:
            adjustments = []
            reasoning = "Simulierte Optimierungsrunde: Wir passen die Parameter leicht an, um das Verhalten zu testen."
            
            for p in miner_output.parameters:
                # Wir verändern den Parameter um +10% seines Werts, innerhalb seiner Schranken
                step = (p.max_bound - p.min_bound) * 0.1
                current_val = current_params.get(p.name, p.value)
                new_val = current_val + step
                if new_val > p.max_bound:
                    new_val = current_val - step
                new_val = max(p.min_bound, min(p.max_bound, new_val))
                
                adjustments.append(ParameterAdjustment(name=p.name, value=new_val))
                
            return OptimizerDecision(
                continue_optimization=True,
                reasoning=reasoning,
                adjustments=adjustments
            )
        else:
            # Nach Runde 2 beenden wir die Optimierung (Konvergenz)
            return OptimizerDecision(
                continue_optimization=False,
                reasoning="Simulierte Konvergenz erreicht: Die Performance-Metrik hat sich stabilisiert und die Invariantentests sind bestanden.",
                adjustments=[]
            )
