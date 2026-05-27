# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Ensure correct pathing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.config import get_client, MinerOutput
from vectornaut.evaluation import (
    build_eval_trace,
    evaluate_run_output,
    list_eval_records,
    load_eval_record,
    load_history_record,
    make_eval_run_id,
    persist_eval_record,
    utcish_now,
)
from vectornaut.api.history_routes import router as history_router
from vectornaut.pipeline import PipelineRunRequest, run_pipeline

# Load environment vars
if load_dotenv:
    load_dotenv()

app = FastAPI(
    title="Vectornaut Omni API",
    description="Backend API services orchestrating Miner, Auditor, and Dynamic Solver stages."
)
app.include_router(history_router)

class RunRequest(PipelineRunRequest):
    previous_miner_output: Optional[MinerOutput] = None

class EvalRunRequest(BaseModel):
    name: Optional[str] = None
    query: str
    epochs: int = 80
    is_mock: bool = True
    override_parameters: Optional[Dict[str, float]] = None
    previous_miner_output: Optional[MinerOutput] = None
    max_optimization_rounds: int = 2
    criteria: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class EvalBatchRequest(BaseModel):
    cases: List[EvalRunRequest]
    stop_on_failure: bool = False

class EvalJudgeRequest(BaseModel):
    run: Dict[str, Any]
    criteria: Dict[str, Any] = Field(default_factory=dict)

class DebugReplayRequest(BaseModel):
    history_file: Optional[str] = None
    eval_run_id: Optional[str] = None
    run: Optional[Dict[str, Any]] = None
    rerun: bool = False
    criteria: Dict[str, Any] = Field(default_factory=dict)
    epochs: Optional[int] = None
    is_mock: Optional[bool] = None

@app.post("/api/run")
async def run_discovery_loop(req: RunRequest):
    try:
        response_data = run_pipeline(req)
        archive_run_data(response_data)
        return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    current_run: Optional[Dict[str, Any]] = None
    is_mock: bool = False

class ParameterProposal(BaseModel):
    name: str = Field(description="Name des Parameters, der optimiert oder verändert werden soll (z. B. 'slip_length').")
    value: float = Field(description="Der neue vorgeschlagene numerische Wert für diesen Parameter.")

class GeminiChatResponse(BaseModel):
    reply: str = Field(description="The detailed conversational reply in German (du/dir address) explaining the physical results, answering the user's questions, or explaining the proposed parameter optimization.")
    suggested_params: List[ParameterProposal] = Field(description="List of proposed parameter updates if optimizing or modifying parameters, otherwise an empty list.", default_factory=list)

SYSTEM_INSTRUCTIONS = """
Du bist der bionische Assistenz-Bot von Vectornaut, einem autonomen System für Material- und Oberflächen-Design.
Deine Aufgabe ist es, dem Benutzer die Simulationsergebnisse und Testergebnisse zu erklären, physikalische Fragen zu beantworten, und Vorschläge zur Optimierung der Parameter zu machen.

Du hast vollen Zugriff auf die Testergebnisse und Details des aktuellen Simulationslaufs im `current_run` Objekt. Nutze diese Daten aktiv!

WICHTIGE STRUKTURIERUNGS- UND FORMATIERUNGS-ANWEISUNGEN (Vermeidung von Textwüsten):
1. STRIKTE KÜRZE & STRUKTUR:
   - Schreibe NIEMALS lange, dichte Fließtext-Absätze.
   - Jeder Absatz darf maximal 2 bis 3 Sätze lang sein. Danach MUSS eine Leerzeile folgen.
   - Nutze Markdown-Überschriften (z.B. ### Physikalischer Hintergrund oder ### Empfohlene Fertigung), um Abschnitte visuell voneinander abzugrenzen.
   - Verwende Aufzählungszeichen (Bullet Points) für Listen, Vergleiche oder Schritte.

2. GLEICHUNGEN & FORMELN AUSLAGERN:
   - Schreibe mathematische Gleichungen, physikalische Formeln oder Definitionen von Variablen NIEMALS im Fließtext.
   - Jede Formel (auch kurze wie B = b / h = 0.2 oder u(1) = v_ski - slip_ratio * du_dy(1)) MUSS in einer eigenen Zeile als Markdown-Codeblock (mit ```) stehen.
   - Beispiel:
     Die Reibungskraft sinkt umgekehrt proportional zum Gleitverhältnis:
     ```
     F_friction ~ 1 / (1 + B)
     ```
     wobei B das Gleitverhältnis ist:
     ```
     B = λ / h = 0.2
     ```

3. SPRACHSTIL & ADRESSIERUNG:
   - Antworte auf Deutsch und verwende konsequent die Anrede per Du ("du", "dir", "dein").
   - Erkläre komplexe physikalische Zusammenhänge verständlich und präzise.

4. PARAMETER-OPTIMIERUNG:
   - Wenn der Benutzer eine Optimierung wünscht, schlage verbesserte Werte für die Parameter vor, die sich innerhalb der geprüften min/max Bounds befinden. Trage diese geänderten Parameter in das Feld 'suggested_params' des Antwort-Schemas ein.
   - Wenn du keine Parameteränderungen vorschlägst, lasse das Feld 'suggested_params' leer.

5. SPRACHLICHE KORREKTHEIT UND UMLAUTE:
   - Verwende im Text unbedingt die korrekten deutschen Umlaute (ä, ö, ü) und das Eszett (ß).
   - Ersetze Umlaute NIEMALS durch Sonderzeichen oder Symbole wie das Dollar-Zeichen ($) (schreibe z. B. immer "Erklärung" statt "Erkl$rung", "ermöglicht" statt "erm$glicht", "über" statt "$ber", "großer" statt "gro$er").
"""

@app.post("/api/chat")
async def chat_with_assistant(req: ChatRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        effective_mock = req.is_mock
        if not api_key and not effective_mock:
            effective_mock = True

        if effective_mock:
            design_name = "Shark-Skin"
            if req.current_run and req.current_run.get("miner"):
                design_name = req.current_run["miner"].get("design_name", "Shark-Skin")

            # Extract parameters and simulation metrics dynamically from context if present
            slip_length = 0.00002
            film_thickness = 0.00001
            viscosity = 0.00179
            gain_pct = 66.67
            
            if req.current_run:
                # Read from miner parameters
                miner_params = req.current_run.get("miner", {}).get("parameters", [])
                for p in miner_params:
                    if p.get("name") == "slip_length":
                        slip_length = p.get("value", slip_length)
                    elif p.get("name") == "film_thickness":
                        film_thickness = p.get("value", film_thickness)
                    elif p.get("name") == "viscosity":
                        viscosity = p.get("value", viscosity)
                
                # Check for auditor overrides
                audited_params = req.current_run.get("auditor", {}).get("audited_parameters_dict", {})
                if "slip_length" in audited_params:
                    slip_length = audited_params["slip_length"]
                if "film_thickness" in audited_params:
                    film_thickness = audited_params["film_thickness"]
                if "viscosity" in audited_params:
                    viscosity = audited_params["viscosity"]
                
                # Read simulator gain
                sim_data = req.current_run.get("simulator", {})
                gain_pct = sim_data.get("performance_gain_pct", gain_pct)

            query_lower = req.message.lower()
            if "opti" in query_lower or "verbesser" in query_lower or "reduzier" in query_lower:
                if "plastron" in design_name.lower() or "ski" in design_name.lower():
                    # Check if already optimized
                    if abs(slip_length - 0.00004) < 1e-6 and abs(film_thickness - 0.000005) < 1e-6:
                        reply = (
                            f"Dieses Design ist bereits maximal optimiert!\n\n"
                            f"### Optimierte Parameter\n\n"
                            f"* Slip-Länge (`slip_length`):\n"
                            f"  ```\n"
                            f"  λ = {slip_length*1e6:.1f} µm\n"
                            f"  ```\n"
                            f"* Filmdicke (`film_thickness`):\n"
                            f"  ```\n"
                            f"  h = {film_thickness*1e6:.1f} µm\n"
                            f"  ```\n\n"
                            f"Damit erzielen wir bereits eine hervorragende Reibungsreduktion von **{gain_pct:.2f}%**.\n\n"
                            f"Das bedeutet, die Reibung wurde auf **1/9** der glatten Referenz reduziert."
                        )
                        suggested_params = {}
                    else:
                        reply = (
                            f"Um den Gleitwiderstand der PlastronGlide-Skibasis weiter zu minimieren, empfehle ich eine Erhöhung der Slip-Länge und eine Reduktion der Wasserfilmdicke.\n\n"
                            f"### Vorgeschlagene Parameter-Änderungen\n\n"
                            f"* Erhöhung von `slip_length` auf:\n"
                            f"  ```\n"
                            f"  λ = 40 µm (0.00004 m)\n"
                            f"  ```\n"
                            f"* Verringerung von `film_thickness` auf:\n"
                            f"  ```\n"
                            f"  h = 5 µm (0.000005 m)\n"
                            f"  ```\n\n"
                            f"### Physikalischer Hintergrund\n\n"
                            f"Das Gleitverhältnis `B` steigt dadurch von dem vorherigen Wert `{slip_length/film_thickness:.1f}` auf:\n"
                            f"```\n"
                            f"B = λ / h = 8\n"
                            f"```\n\n"
                            f"Dies ermöglicht theoretisch eine Reibungsreduktion von fast **88.89%** (Reibung auf **1/9** der Referenz reduziert).\n\n"
                            f"Klicke unten auf **'Apply & Run Next Optimization Round'**, um diese optimierten Werte zu simulieren."
                        )
                        suggested_params = {"slip_length": 0.00004, "film_thickness": 0.000005}
                else:
                    reply = (
                        "Für eine weitere Optimierung dieses Strömungsprofils empfehle ich, die geometrischen Abmessungen anzupassen.\n\n"
                        "### Empfohlene Vorgehensweise\n\n"
                        "Wir verändern die geometrischen Störungen (z. B. Riblet-Höhe oder Spacing), um die Wirbelbildung noch effektiver aus der viskosen Unterschicht zu verdrängen.\n\n"
                        "Ich habe die Parameter entsprechend angepasst:\n\n"
                        "* Riblet-Höhe / Dicke: Reduziert um 20%\n"
                        "* Riblet-Abstand / Länge: Erhöht um 20%\n\n"
                        "Klicke unten auf **'Apply & Run Next Optimization Round'**, um die neuen Parameter zu testen."
                    )
                    suggested_params = {}
                    if req.current_run and req.current_run.get("miner", {}).get("parameters"):
                        for p in req.current_run["miner"]["parameters"]:
                            if "height" in p["name"] or "thickness" in p["name"]:
                                suggested_params[p["name"]] = p["value"] * 0.8
                            elif "spacing" in p["name"] or "length" in p["name"]:
                                suggested_params[p["name"]] = p["value"] * 1.2
            elif "herstell" in query_lower or "fertig" in query_lower or "produz" in query_lower:
                reply = (
                    "Die Herstellung einer solchen hierarchischen nanostrukturierten Oberfläche kann über verschiedene moderne Verfahren erfolgen:\n\n"
                    "1. **Laser-Direktstrukturierung (DLIP):**\n"
                    "   Ultrakurzpulslaser können hochpräzise periodische Mikro- und Nanostrukturen direkt auf die Skibasis (meist UHMWPE) brennen.\n\n"
                    "2. **Heißprägen:**\n"
                    "   Strukturierte Walzen übertragen das bionische Collembolen-Muster unter Hitze und Druck auf das Belagmaterial.\n\n"
                    "3. **Additive Fertigung (3D-Druck):**\n"
                    "   Hochauflösende stereolithografische Verfahren (SLA) im Mikromaßstab eignen sich hervorragend für Prototypen.\n\n"
                    "Gibt es ein bestimmtes Verfahren, das du für deine Fertigungsplanung bevorzugst?"
                )
                suggested_params = {}
            elif "warum" in query_lower or "wieso" in query_lower or "erklär" in query_lower or "ursache" in query_lower or "effekt" in query_lower or "reduktion" in query_lower or "reibungs" in query_lower or "66" in query_lower or "88" in query_lower or "1/3" in query_lower:
                if "plastron" in design_name.lower() or "ski" in design_name.lower():
                    slip_um = slip_length * 1e6
                    film_um = film_thickness * 1e6
                    ratio_remaining = film_thickness / (film_thickness + slip_length)
                    
                    if abs(ratio_remaining - 1/3) < 1e-3:
                        fraction_str = "1/3"
                        reduction_explanation = "Die Wandschubspannung (Reibung) wird auf genau 1/3 des Wertes einer glatten No-Slip-Wand reduziert. Der Reibungsverlust beträgt folglich nur noch 33.33% der Referenz – was einer Einsparung von exakt 66.67% entspricht!"
                    elif abs(ratio_remaining - 1/9) < 1e-3:
                        fraction_str = "1/9"
                        reduction_explanation = "Die Wandschubspannung (Reibung) wird auf genau 1/9 des Wertes einer glatten No-Slip-Wand reduziert. Der Reibungsverlust beträgt folglich nur noch 11.11% der Referenz – was einer Einsparung von exakt 88.89% entspricht!"
                    else:
                        fraction_str = f"{ratio_remaining:.3f}"
                        reduction_explanation = f"Die Wandschubspannung (Reibung) wird auf {ratio_remaining*100:.2f}% des Wertes einer glatten No-Slip-Wand reduziert (eine Einsparung von exakt {gain_pct:.2f}%)."

                    reply = (
                        f"Die Reibungsreduktion von **{gain_pct:.2f}%** ergibt sich direkt aus der Slip-Länge und der Wasserfilmdicke.\n\n"
                        f"### Physikalische Parameter\n\n"
                        f"* Slip-Länge:\n"
                        f"  ```\n"
                        f"  λ = {slip_um:.1f} µm\n"
                        f"  ```\n"
                        f"* Wasserfilmdicke:\n"
                        f"  ```\n"
                        f"  h = {film_um:.1f} µm\n"
                        f"  ```\n\n"
                        f"### Formel für die Reibungskraft (Schubspannung)\n\n"
                        f"Die Reibungskraft sinkt umgekehrt proportional zu `(1 + B)`:\n"
                        f"```\n"
                        f"F_friction ~ 1 / (1 + B)\n"
                        f"```\n\n"
                        f"wobei das dimensionslose Gleitverhältnis (slip_ratio) `B` definiert ist als:\n"
                        f"```\n"
                        f"B = λ / h = {slip_um/film_um:.1f}\n"
                        f"```\n\n"
                        f"### Physikalischer Hintergrund\n\n"
                        f"Da der bionische Belag eine Slip-Länge `λ` aufweist, ist der wirksame Geschwindigkeitsgradient an der Wand um einen Faktor reduziert:\n"
                        f"```\n"
                        f"Faktor = h / (h + λ)\n"
                        f"```\n\n"
                        f"In diesem Fall entspricht das einem Verhältnis von:\n"
                        f"```\n"
                        f"{film_um:.1f} / ({film_um:.1f} + {slip_um:.1f}) = {fraction_str}\n"
                        f"```\n\n"
                        f"{reduction_explanation}"
                    )
                else:
                    reply = (
                        f"Dieses Ergebnis (eine Effizienz von **{gain_pct:.2f}%**) resultiert aus dem veränderten Geschwindigkeitsgradienten an der Oberfläche.\n\n"
                        f"### Wirkmechanismus\n\n"
                        f"Durch die bionische Struktur wird die Strömung lokal beeinflusst (entweder durch Gleiten an den Kavitäten oder Verdrängen von Scherwirbeln).\n\n"
                        f"Dies führt zu einer messbaren Reduktion der Wandschubspannung im Vergleich zur glatten Referenz."
                    )
                suggested_params = {}
            elif "test" in query_lower or "ergebnis" in query_lower or "daten" in query_lower:
                # Dynamic extraction of results
                ui_meta = req.current_run.get("auditor", {}).get("ui_metadata", {}) if req.current_run else {}
                primary_label = ui_meta.get("primary_metric", {}).get("label", "Bionische Metrik")
                reference_label = ui_meta.get("reference_metric", {}).get("label", "Referenzmetrik")
                gain_label = ui_meta.get("performance_gain", {}).get("label", "Effizienzsteigerung")
                
                sim_data = req.current_run.get("simulator", {}) if req.current_run else {}
                primary_val = sim_data.get("primary_metric_value", sim_data.get("wall_shear_stress_pinn", 0.0))
                reference_val = sim_data.get("reference_metric_value", sim_data.get("wall_shear_stress_analytical", 0.0))
                solver_method = sim_data.get("solver_method", "N/A")
                
                loss_info = ""
                if solver_method == "pinn":
                    loss_info = f"\n* **PINN End-Loss (MSE):** `{sim_data.get('final_loss', 0.0):.4e}` (trainiert für {sim_data.get('epochs_trained', 0)} Epochen)"
                
                reply = (
                    f"Hier sind die Testergebnisse und Simulationsdaten für den Entwurf **{design_name}**:\n\n"
                    f"### Simulationsübersicht\n"
                    f"* Solver-Methode: `{solver_method.upper()}`\n"
                    f"* {gain_label}: **{gain_pct:.2f}%**\n\n"
                    f"### Kennzahlen\n"
                    f"* {primary_label} (Bionisch): `{primary_val:.6f}`\n"
                    f"* {reference_label} (Referenz): `{reference_val:.6f}`\n"
                    f"* Relative Abweichung (Fehler): `{sim_data.get('relative_error', 0.0):.4e}`\n"
                    f"{loss_info}\n\n"
                    f"Möchtest du, dass ich diese Testergebnisse physikalisch interpretiere oder eine Parameter-Optimierung vorschlage?"
                )
                suggested_params = {}
            else:
                reply = (
                    f"Hallo! Ich habe die Simulationsergebnisse für den Entwurf **{design_name}** analysiert.\n\n"
                    f"### Wichtigstes Ergebnis\n"
                    f"* Reibungsreduktion: **{gain_pct:.2f}%**\n\n"
                    f"Du kannst mich fragen, wie sich diese Struktur physikalisch verhält, wie man sie fertigen kann oder wir können direkt eine Optimierung der Parameter durchführen."
                )
                suggested_params = {}
            
            return JSONResponse(content={"reply": reply, "suggested_params": suggested_params}, media_type="application/json; charset=utf-8")

        else:
            from google.genai import types

            client = get_client()
            
            history_str = ""
            for msg in req.history:
                role_label = "Benutzer" if msg.role == "user" else "Assistent"
                history_str += f"{role_label}: {msg.content}\n"
                
            context_prompt = f"""
            Hier ist der Kontext des aktuellen Simulationslaufs:
            {json.dumps(req.current_run, indent=2, ensure_ascii=False)}
            
            Bitte beantworte die folgende Benutzeranfrage unter Berücksichtigung dieses Kontextes und des bisherigen Verlaufs.
            Benutzeranfrage: "{req.message}"
            """
            
            full_prompt = f"{SYSTEM_INSTRUCTIONS}\n\n"
            if history_str:
                full_prompt += f"Bisheriger Chatverlauf:\n{history_str}\n"
            full_prompt += context_prompt
            
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiChatResponse,
                )
            )
            parsed_res = response.parsed
            suggested_dict = {}
            if parsed_res and parsed_res.suggested_params:
                for prop in parsed_res.suggested_params:
                    suggested_dict[prop.name] = prop.value
            
            reply_text = parsed_res.reply if parsed_res else "Keine Antwort erhalten."
            # Clean double escaped newlines
            reply_text = reply_text.replace("\\n", "\n")
            
            return JSONResponse(content={
                "reply": reply_text,
                "suggested_params": suggested_dict
            }, media_type="application/json; charset=utf-8")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def _execute_eval_run(req: EvalRunRequest) -> Dict[str, Any]:
    run_id = make_eval_run_id()
    started_at = utcish_now()
    start_perf = time.perf_counter()
    output_data: Optional[Dict[str, Any]] = None
    status = "ok"
    error_message: Optional[str] = None

    run_request = PipelineRunRequest(
        query=req.query,
        epochs=req.epochs,
        is_mock=req.is_mock,
        override_parameters=req.override_parameters,
        previous_miner_output=req.previous_miner_output,
        max_optimization_rounds=req.max_optimization_rounds,
    )

    try:
        output_data = run_pipeline(run_request)
        archive_run_data(output_data)
    except Exception as exc:
        status = "failed"
        error_message = str(exc)

    duration_ms = int((time.perf_counter() - start_perf) * 1000)
    finished_at = utcish_now()
    request_data = req.model_dump()
    trace = build_eval_trace(
        run_id=run_id,
        request_data=request_data,
        run_data=output_data,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        status=status,
        error=error_message,
    )
    evaluation = evaluate_run_output(output_data or {}, req.criteria)
    if status == "failed":
        evaluation["passed"] = False

    record = {
        "run_id": run_id,
        "name": req.name,
        "tags": req.tags,
        "status": status,
        "trace": trace,
        "evaluation": evaluation,
        "output": output_data,
        "error": error_message,
    }
    record["storage_path"] = persist_eval_record(record)
    return record

@app.post("/api/eval/run")
async def eval_run(req: EvalRunRequest):
    record = await _execute_eval_run(req)
    return JSONResponse(content=record, media_type="application/json; charset=utf-8")

@app.post("/api/eval/batch")
async def eval_batch(req: EvalBatchRequest):
    records = []
    for case in req.cases:
        record = await _execute_eval_run(case)
        records.append(record)
        if req.stop_on_failure and not record.get("evaluation", {}).get("passed", False):
            break

    passed_count = sum(1 for record in records if record.get("evaluation", {}).get("passed", False))
    response = {
        "success": passed_count == len(records) and len(records) == len(req.cases),
        "requested": len(req.cases),
        "executed": len(records),
        "passed": passed_count,
        "failed": len(records) - passed_count,
        "records": records,
    }
    return JSONResponse(content=response, media_type="application/json; charset=utf-8")

@app.post("/api/eval/judge")
async def eval_judge(req: EvalJudgeRequest):
    evaluation = evaluate_run_output(req.run, req.criteria)
    return JSONResponse(content=evaluation, media_type="application/json; charset=utf-8")

@app.get("/api/eval/runs")
async def list_eval_runs(limit: int = 50):
    records = list_eval_records(limit=limit)
    return JSONResponse(content={"runs": records}, media_type="application/json; charset=utf-8")

@app.get("/api/eval/runs/{run_id}")
async def get_eval_run(run_id: str):
    try:
        record = load_eval_record(run_id)
        return JSONResponse(content=record, media_type="application/json; charset=utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.post("/api/debug/replay")
async def debug_replay(req: DebugReplayRequest):
    try:
        source_data: Optional[Dict[str, Any]] = None
        source_label = "inline"

        if req.run is not None:
            source_data = req.run
        elif req.eval_run_id:
            record = load_eval_record(req.eval_run_id)
            source_label = req.eval_run_id
            source_data = record.get("output")
            if source_data is None:
                source_data = record
        elif req.history_file:
            source_label = req.history_file
            source_data = load_history_record(req.history_file)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: run, eval_run_id, or history_file.")

        if not req.rerun:
            evaluation = evaluate_run_output(source_data or {}, req.criteria)
            replay_run_id = make_eval_run_id()
            replay_record = {
                "run_id": replay_run_id,
                "name": f"replay:{source_label}",
                "status": "replayed",
                "trace": build_eval_trace(
                    run_id=replay_run_id,
                    request_data={"source": source_label, "rerun": False},
                    run_data=source_data,
                    started_at=utcish_now(),
                    finished_at=utcish_now(),
                    duration_ms=0,
                    status="replayed",
                ),
                "evaluation": evaluation,
                "output": source_data,
                "error": None,
            }
            replay_record["storage_path"] = persist_eval_record(replay_record)
            return JSONResponse(content=replay_record, media_type="application/json; charset=utf-8")

        query = None
        if source_data:
            query = source_data.get("query")
        if not query and req.eval_run_id:
            record = load_eval_record(req.eval_run_id)
            query = get_nested_request_query(record)
        if not query:
            raise HTTPException(status_code=400, detail="Cannot rerun this record because it does not contain the original query.")

        eval_req = EvalRunRequest(
            name=f"rerun:{source_label}",
            query=query,
            epochs=req.epochs or int(source_data.get("epochs", 80) if source_data else 80),
            is_mock=req.is_mock if req.is_mock is not None else bool(source_data.get("is_mock", True) if source_data else True),
            criteria=req.criteria,
            tags=["debug-replay"],
        )
        record = await _execute_eval_run(eval_req)
        return JSONResponse(content=record, media_type="application/json; charset=utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

def get_nested_request_query(record: Dict[str, Any]) -> Optional[str]:
    trace = record.get("trace", {})
    request = trace.get("request", {}) if isinstance(trace, dict) else {}
    query = request.get("query") if isinstance(request, dict) else None
    return query if isinstance(query, str) and query.strip() else None

def archive_run_data(response_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    try:
        os.makedirs("history", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        design_name = response_data.get("miner", {}).get("design_name") or "unknown_design"
        slug = re.sub(r'[^a-zA-Z0-9_]', '', design_name.lower().replace(" ", "_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_content = generate_markdown_report_content(response_data, timestamp)
        response_data["report_md"] = md_content

        md_filename = f"report_{timestamp}_{slug}.md"
        md_path = os.path.join("reports", md_filename)
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(md_content)

        json_filename = f"run_{timestamp}_{slug}.json"
        json_path = os.path.join("history", json_filename)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(response_data, jf, indent=4, ensure_ascii=False)

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
            reasoning = run.get("optimizer_reasoning", "Konvergenz erreicht oder Limit erreicht.").replace("\n", " ").strip()
            md.append(f"| {r_num} | {params_str} | `{coeff_val:.6f}` | **{gain:.2f}%** | {val_status} | {reasoning} |")

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

# Mount static folder for frontend UI (will be served at root "/")
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    print("[*] Launching Vectornaut 2.0 Web Server on http://0.0.0.0:8080 ...")
    uvicorn.run("web_server:app", host="0.0.0.0", port=8080, reload=False)
