# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vectornaut.config import get_client

router = APIRouter(prefix="/api", tags=["chat"])
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

@router.post("/chat")
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

