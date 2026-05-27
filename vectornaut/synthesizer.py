# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from .config import get_client, MinerOutput, AuditorOutput, SimulatorOutput

class SynthesisReport(BaseModel):
    executive_summary: str = Field(description="Eine leicht verständliche Zusammenfassung der bionischen Idee, der physikalischen Funktionsweise und der verwendeten Materialien (auf Deutsch, informelles 'du').")
    pros_and_cons: str = Field(description="Eine Gegenüberstellung der Vorteile und Nachteile (bzw. Herausforderungen) dieses Entwurfs im Vergleich zu bestehenden, konventionellen Lösungen (auf Deutsch, informelles 'du').")
    mechanical_limits: str = Field(description="Detaillierte Analyse der mechanischen Belastungsgrenzen, Stabilität unter Last und Ausfallkriterien (auf Deutsch, informelles 'du').")
    manufacturing_methods: str = Field(description="Empfohlene Fertigungstechniken und Skalierungsmethoden für die Realisierung der Mikro- oder Nanostrukturen (auf Deutsch, informelles 'du').")
    cost_estimation: str = Field(description="Grobe Kostenschätzung für Prototyping, Werkzeugkosten und Massenproduktion (auf Deutsch, informelles 'du').")
    validation_experiments: str = Field(description="Vorschläge für konkrete Labor- und Feldtests zur Validierung der bionischen Leistung (auf Deutsch, informelles 'du').")
    industry_partners: str = Field(description="Mögliche Industriebranchen, Partnertypen oder reale Lösungsanbieter für die Umsetzung (auf Deutsch, informelles 'du').")

class Synthesizer:
    def __init__(self, client=None):
        self.client = client

    def generate_synthesis(
        self,
        miner_output: MinerOutput,
        auditor_output: AuditorOutput,
        simulator_output: SimulatorOutput,
        user_query: str
    ) -> SynthesisReport:
        """
        Generiert einen umfassenden kommerziellen und praktischen Synthese-Report.
        """
        from google.genai import types

        prompt = f"""
        Du bist der führende bionische Material- und Produktionsingenieur von Vectornaut. Deine Aufgabe ist es, einen fertigen, simulierten bionischen Materialentwurf kommerziell und praktisch zu bewerten.
        
        ### Benutzer-Forschungsaufgabe
        "{user_query}"

        ### Bionisches Konzept
        Name: {miner_output.design_name}
        Natürliches Vorbild: {miner_output.inspiration_source}
        Physikalischer Mechanismus: {miner_output.physical_mechanism}
        Parameter (optimiert): {auditor_output.audited_parameters_dict}
        Simulations-Koeffizient: {auditor_output.simulation_coefficient}

        ### Simulations- und Testergebnisse
        Lösungsmethode: {simulator_output.solver_method}
        Effizienz (Performance Gain): {simulator_output.performance_gain_pct:.2f}%
        Auditor Notizen (Belastung): {auditor_output.audit_notes}

        Erstelle basierend auf diesen Daten einen detaillierten Bericht, der für einen Industriepartner die praktische Umsetzung beschreibt. Halte dich an den informellen 'du'-Stil auf Deutsch. Liefere präzise, ingenieurwissenschaftliche Beschreibungen für jedes Feld im geforderten JSON-Schema.
        """

        client = self.client or get_client()
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SynthesisReport,
            )
        )
        return response.parsed

    def mock_generate_synthesis(self) -> SynthesisReport:
        """
        Mock-Implementierung für den Offline- und Testmodus.
        """
        return SynthesisReport(
            executive_summary="Executive Summary: Diese bionische Struktur nutzt die optimierten Oberflächenstrukturen, um Reibungsverluste zu minimieren oder den Stofftransport zu maximieren. Durch die gezielte Wahl von Elastomeren und strukturierten Oberflächen wird eine hohe Effizienz erreicht.",
            pros_and_cons="**Vorteile:**\n* Deutliche Leistungssteigerung gegenüber Standardmaterialien.\n* Nachhaltige, physikalische Wirkweise ohne chemische Zusätze.\n\n**Nachteile/Herausforderungen:**\n* Höhere Werkzeugkosten für die Mikrostrukturierung.\n* Eventuell begrenzte mechanische Abriebfestigkeit der filigranen Strukturen.",
            mechanical_limits="Simulierte Belastungsgrenzen: Die Struktur hält einer theoretischen Flächenpressung von bis zu 12 MPa stand, bevor elastisches Knicken der Mikro-Prismen auftritt.",
            manufacturing_methods="Empfohlene Fertigung: Großflächiges Heißprägen mittels strukturierter Walzen oder direkte Ultrakurzpulslaser-Strukturierung (DLIP).",
            cost_estimation="Prototyp-Entwicklung: ca. 15.000 € für die Prägewalze. Massenproduktion: Aufpreis von ca. 2,50 € pro Quadratmeter gegenüber dem unstrukturierten Basismaterial.",
            validation_experiments="Validierung: 1. Tribometer-Verschleißtests im Labor zur Bestimmung des Gleitkoeffizienten. 2. Haltbarkeitstests unter wechselnden Zyklen.",
            industry_partners="Mögliche Partner: Hersteller von Funktionstextilien, Sportgerätehersteller oder Spezialbeschichtungsfirmen in der Kunststofftechnik."
        )
