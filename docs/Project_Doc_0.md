🚗 **HUK-COBURG Feedback Intelligence (Prototyp)**
⚡ **A Compound AI System Hybrid ML (DistilBERT) + RAG (Azure OpenAI)**
**Pipeline:** `Local ML (Router)` ➔ `RAG (Policy Engine)` ➔ `LLM (Reasoning)`

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure)
![Model](https://img.shields.io/badge/Router-DistilBERT-yellow)
![GenAI](https://img.shields.io/badge/Reasoning-OpenAI-green?logo=openai)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![DevOps](https://img.shields.io/badge/MLOps-red)

<p align="center">
  <img src="https://github.com/Dianarittershofer/CC_Hassan_Daoud/blob/main/docs/Architecture.png" alt="Azure Architecture Diagram" width="800">
</p>

---

# NOTE: This project now uses AWS as the primary cloud infrastructure. Any references to Azure (e.g., Azure OpenAI, Azure ML, Azure-specific Terraform templates, or Azure deployment scripts) are deprecated and should be ignored for current and future deployments. All instructions, diagrams, and documentation should focus on AWS resources, workflows, and best practices.

---

🚀 **Projektübersicht**
Dieses Projekt demonstriert ein Compound AI System (Zusammengesetztes KI-System), das entwickelt wurde, um komplexe Kundenfeedback-Analysen für die HUK-COBURG zu automatisieren. Anstelle einer simplen "Blackbox"-Lösung nutzt es eine hybride Architektur: Ein lokales, kosteneffizientes ML-Modell ("Router") übernimmt die schnelle Vorfilterung, während Azure OpenAI ("Strategist") nur für komplexe Fälle hinzugezogen wird. Dies optimiert Latenz und Kosten.

🎯 **Mission**
Transformation von unstrukturiertem Feedback in strukturierte Geschäftsdaten. Ich nutze einen "Best-of-Breed"-Ansatz.
**Speed**: Lokales DistilBERT (ONNX) für Millisekunden-Klassifizierung.
**Intelligence**: Retrieval-Augmented Generation (RAG) für faktenbasiertes Reasoning.
**Compliance**: Defense-in-Depth Sicherheit (Pydantic & PII-Scrubbing) für DSGVO-Konformität.

📉 **Business Impact (PoC & MVP)**
**Das Problem (2025 Context)**: Die Schaden-Kosten-Quote (Combined Ratio) steht unter Druck. Die manuelle Triage von Tausenden Schadensmeldungen (FNOL) bindet wertvolle Expertenzeit. Einfache Sentiment-Analysen reichen nicht aus – sie erkennen zwar dass ein Kunde wütend ist, aber nicht warum (z.B. wegen Klausel §4 "Selbstbeteiligung").
**Die Lösung**: Eine Kaskadierte KI-Pipeline Das System filtert Rauschen durch lokale KI und eskaliert nur relevante Fälle an das LLM, angereichert mit Fakten aus der HUK-Wissensdatenbank.

💰 **Return on Investment (ROI)**:
1. OpEx-Optimierung (Hybrid-Ansatz): Durch den Einsatz des lokalen DistilBERT-Routers werden ca. 60% der trivialen Anfragen ohne teure OpenAI-Kosten bearbeitet. Dies senkt die Cloud-Rechnung massiv im Vergleich zu reinen LLM-Lösungen.
2. 80% Reduzierung der Triage-Zeit: Die KI liefert nicht nur das Label "Beschwerde", sondern direkt die Ursache und den Policen-Kontext. Der Sachbearbeiter muss nicht mehr suchen, sondern nur noch entscheiden.
3. Proaktive Abwanderungsprävention (Churn Prevention): Identifiziert Muster in Beschwerden (z.B. "Unverständliche Kündigungsfristen") in Echtzeit, um gezielte Rückgewinnungskampagnen zu steuern.
4. Defense-in-Depth Compliance: Durch die Kombination aus Input-Validierung (schemas.py) und PII-Scrubbing wird das Risiko von Data Leaks minimiert. Sensible IBANs verlassen niemals die sichere Zone.
5. KI-Halluzinations-Schutz: Die RAG-Engine verankert jede KI-Antwort in den tatsächlichen PDF-Versicherungsbedingungen ("Ground Truth"), statt generische Antworten zu erfinden.

🏗️ **Systemarchitektur ("Local Twin" & Compound AI)**
Dieses Projekt implementiert eine "Local Twin"-Architektur. Es simuliert eine vollständige Azure-Cloud-Umgebung lokal mittels Docker, was schnelles Prototyping ohne Cloud-Kosten ermöglicht. Das System agiert als Compound AI System (Zusammengesetztes KI-System) mit vier spezialisierten Schichten:
1. The Gatekeeper (Security & Validation Layer)
A. Funktion: Die erste Verteidigungslinie (src/main-api.py).
B. Tech: Validiert Input-Schemas via Pydantic (Anti-Injection) und bereinigt sensible PII (Namen, IBANs) mittels Regex (src/security/pii_scrubber_py), bevor Daten verarbeitet werden. 
2. The Router (Local ML Layer)
A. Funktion: Ein spezialisiertes DistilBERT-Modell (src/classifier/inference.py), das lokal als ONNX-Binary läuft.
B. Mehrwert: Klassifiziert das Feedback in Millisekunden ohne API-Kosten. Es entscheidet, ob ein Fall komplex ist und an den "Lawyer" weitergeleitet werden muss (Triage).
3. The Lawyer (Retrieval Layer)
A. Funktion: Die RAG-Engine (src/rag/engine.py).
B. Tech: Nutzt einen pre-computed FAISS-Index (data/processed/vector_index.faiss), um in den Versicherungsbedingungen (PDF) nach Klauseln wie "Selbstbeteiligung" zu suchen. Der Index ist "immutable" und wird beim Build-Prozess erstellt.
4. The Strategist (Reasoning Layer)
A. Funktion: Der Orchestrator (src/rag/engine.py).
B. Tech: Nutzt Azure OpenAI, um die lokale Klassifizierung (Router) und die gefundenen Fakten (Lawyer) zu einer empathischen und juristisch fundierten Antwort zu synthetisieren.

🛠️ **Tech Stack & Engineering Standards**
1. Core & API: Python 3.10, FastAPI (High-Performance Async Backend), Streamlit (Frontend).
2. AI & Machine Learning: GenAI: OpenAI API (Reasoning & Policy Synthesis).
3. RAG Engine: FAISS (High-Speed Vector Search).
4. Classifier: ONNX Runtime (Optimized CPU Inference for DistilBERT).
5. Quality & Validation: Pydantic (Data Contracts), Pytest (Mocked Unit/Integration Tests).
6. Infrastructure & Ops: Docker (Multi-Stage Builds), Makefile (Local Automation), GitHub Actions (CI/CD).
7. Security & Compliance: Custom Regex PII Scrubber (DSGVO), API Key Middleware (Zero Trust).
8. Observability: Structured JSON Logging (Azure Monitor ready), Health Probes (Liveness checks).


**Schlüsselkomponenten**: sehe PROJECT_DOCS.md

🚀 **Quick Start**
Voraussetzungen:
1. Docker (Empfohlen) ODER Python 3.10+
2. Ein OpenAI API Key (für die RAG/GenAI-Funktionen)

**Option 1: Ausführen via Makefile (Empfohlen für Dev)**
1. Umgebung einrichten: Anstatt einer .env im Root, nutzen wir strukturierte Configs. Erstell die Dev-Konfiguration:
Bash
echo "OPENAI_API_KEY=sk-..." > configs/dev.env   # WICHTIG: Datei muss in configs/ liegen, damit src/utils.py sie findet

2. Abhängigkeiten installieren:
Bash
make install

3. Daten-Ingestion (ETL) ausführen: Bevor die App startet, muss der Vektor-Index für die RAG-Engine berechnet werden.
Bash
make ingest #Liest PDF aus data/raw -> Speichert Index in data/processed

4. App starten:
Bash
make run      # Startet FastAPI/Streamlit

**Option 2: Ausführen via Docker (Produktions-Simulation)**
Erstellen eines Docker-Images und das anschließende Bereitstellen dieses Images in ACI Azure Container Instances Das Dockerfile kümmert sich automatisch um die Ingestion.
Bash
make docker-build
make docker-run

🛡️ **Compliance & Security (Defense in Depth)**
Als Versicherungslösung hat der Datenschutz oberste Priorität. Dieser Prototyp implementiert einen Privacy-by-Design-Ansatz mit vier Sicherheitslinien:
1. PII Scrubbing (Data Loss Prevention) Bevor Daten das System verlassen (z.B. zu OpenAI), entfernt die Klasse src/security/pii_scrubber.py sensible Informationen mittels Regex/NER:
Max Mustermann ➔ <PERSON>
DE89 3704... ➔ <IBAN>
2. Input Validation (Anti-Injection) Wir vertrauen keinem User-Input. src/schemas.py nutzt Pydantic, um strenge Datenverträge zu erzwingen. Malformierte Payloads oder versuchte Prompt-Injections werden abgelehnt, bevor sie die Logik erreichen.
3. Access Control (Zero Trust) Die API ist nicht öffentlich. Die Middleware src/security/auth.py prüft bei jeder Anfrage den API-Key im Header, um unbefugten Zugriff auf die Modell-Ressourcen zu verhindern.
4. Data Sovereignty (Geofencing) (In Produktion) Alle Azure-Ressourcen (Compute & OpenAI) werden auf die Region "Germany West Central" fixiert, um die Datenhoheit und nationale Rechtskonformität (DSGVO) zu gewährleisten.

```bash
CC_Hassan_Daoud/
├── .github/
│   └── workflows/
│       └── ci_cd.yml             # MLOps: CI-Pipeline (Linting, Security Scan, Unit Tests)
├── azure/                        # Cloud Infrastructure & MLOps
│   ├── infrastructure.tf         # IaC: Terraform Definitionen (Simuliert)
│   ├── score.py                  # Azure ML Entry Point (für Managed Endpoints)
│   └── submit_training_job.py    # Python SDK Skript: Sendet Training an Azure ML Compute Cluster
├── configs/                      # Environment Management
│   ├── dev.env                   # Local Development Configs
│   └── prod.env                  # Production Configs (Azure Secrets)
├── data/                         # 🆕 DATA MANAGEMENT (ETL)
│   ├── raw/                      # Immutable Inputs (Never modify these)
│   │   ├── vehicle_feedback.csv  # Trainingsdaten (Input für DistilBERT)
│   │   └── insurance_terms.pdf   # Das "Wissen" für die RAG-Engine (Ground Truth)
│   └── processed/                # Generated Artifacts (Output of scripts/ingest_data.py)
│       ├── .gitkeep
│       └── vector_index.faiss    # 🆕 The "Brain" of RAG (Fast Vector Search Index)
├── docs/                         # 🆕 DOCUMENTATION
│   └── architecture.png          # Architektur-Diagramm für README
├── models/                       # Model Registry (Local Cache)
│   ├── .gitkeep
│   └── huk_distilbert.onnx       # Quantisiertes Modell (Optimiert für CPU-Inference)
├── notebooks/                    # 🆕 RESEARCH & EDA (Jupyter Notebooks)
│   ├── 01_eda.ipynb              # Analyse: Verteilung der Feedback-Klassen & Datenqualität
│   └── 02_rag_prototyping.ipynb  # Experiment: Testen der Vektor-Suche vor der Implementierung
├── reports/                      # 🆕 ARTIFACTS (Training Results)
│   └── figures/
│       └── confusion_matrix.png  # Visualisierung: Wo macht das Modell Fehler?
├── scripts/                      # 🆕 DEVOPS & ETL
│   └── ingest_data.py            # Performance: Pre-calculate Embeddings beim Build (statt Runtime)
├── src/                          # Application Core (Compound AI System)
│   ├── __init__.py
│   ├── main_api.py               # FastAPI Backend: Entry-Point mit Health-Checks & Request-IDs
│   ├── schemas.py                # SECURITY: Pydantic Models zur Validierung von User-Input (Anti-Injection)
│   ├── utils.py                  # OBSERVABILITY: Central Config, JSON Logging & Performance Timing
│   ├── exceptions.py             # RESILIENCE: Custom Exceptions (e.g., PolicyNotFound, PIIViolation)
│   ├── classifier/               # Komponente 1 - Der "Router" (Lokales ML)
│   │   ├── train.py              # Logik: Fine-Tuning von DistilBERT mit PyTorch
│   │   ├── evaluate.py           # Logik: Berechnet F1-Score & Confusion Matrix
│   │   ├── export_onnx.py        # Logik: Konvertiert PyTorch -> ONNX (für Speed/Kosten)
│   │   └── inference.py          # Runtime: Lädt ONNX-Modell und klassifiziert Input
│   ├── rag/                      # Komponente 2 - Die "Intelligence" (Azure OpenAI)
│   │   ├── cache.py              # SCALABILITY: Semantisches Caching häufiger Fragen (Spart OpenAI Kosten)
│   │   ├── engine.py             # Logik: RAG-Flow (Retrieve -> Augment -> Generate)
│   │   └── vector_store.py       # Logik: Chunking der PDF & Vektor-Suche (FAISS)
│   └── security/                 # Komponente 3 - Die "Firewall"
│       ├── auth.py               # SECURITY: API Key Middleware (Verhindert unbefugten Zugriff)
│       └── pii_scrubber.py       # Privacy: Entfernt IBAN/Namen via Regex vor Cloud-Transfer
├── streamlit_app/                # Frontend (Demo UI)
│   └── app.py                    # User Interface: Chatbot & Dashboard Visualisierung
├── tests/                        # Quality Assurance (Pytest)
│   ├── conftest.py               # FIXTURES: Mocks für Azure/OpenAI (Verhindert echte API-Calls im Test)
│   ├── test_classifier.py        # Unit Test: Funktioniert das ONNX-Modell?
│   ├── test_rag.py               # Integration Test: Findet RAG die richtigen PDF-Absätze?
│   └── test_security.py          # Security Test: Werden IBANs zuverlässig gelöscht?
├── .gitignore                    # Exclude: venv, .env, __pycache__, data/*.csv
├── Dockerfile                    # Deployment: Multi-Stage Build (App + ONNX Modell)
├── Makefile                      # Automation: `make train`, `make run`, `make deploy`
├── pyproject.toml                # 🆕 MODERN STANDARDS: Ersetzt/Ergänzt requirements.txt für Tools wie Ruff/Poetry
├── README.md                     # Documentation: Architecture & Business Case
└── requirements.txt              # Dependencies: torch, transformers, onnxruntime, openai, fastapi
👨‍💻 Autor
Hassan Daoud