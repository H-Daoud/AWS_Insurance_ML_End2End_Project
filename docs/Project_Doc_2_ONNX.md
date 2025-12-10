# Model Integration and RAG Workflow
**Pipeline:** `Local ML (Router)` ➔ `RAG (Policy Engine)` ➔ `LLM (Reasoning)`

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure)
![Model](https://img.shields.io/badge/Router-DistilBERT-yellow)
![GenAI](https://img.shields.io/badge/Reasoning-OpenAI-green?logo=openai)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![DevOps](https://img.shields.io/badge/MLOps-red)

---

# NOTE: This project now uses AWS as the primary cloud infrastructure. Any references to Azure (e.g., Azure OpenAI, Azure ML, Azure-specific Terraform templates, or Azure deployment scripts) are deprecated and should be ignored for current and future deployments. All instructions, diagrams, and documentation should focus on AWS resources, workflows, and best practices.

<p align="center">
  <img src="https://github.com/Dianarittershofer/CC_Hassan_Daoud/blob/main/docs/Architecture.png" alt="Azure Architecture Diagram" width="400">
</p>

<table>
  <thead>
    <tr>
      <th>Datei / Pfad</th>
      <th>Funktion</th>
      <th>Typ</th>
      <th>Beschreibung der Rolle</th>
      <th>Strategischer Mehrwert (ROI & Business Value)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong><code>.github/workflows/ci_cd.yml</code></strong></td>
      <td>CI/CD Pipeline</td>
      <td>MLOps</td>
      <td>Automatisiert Linting, Security Scans und Unit Tests bei jedem Push. Verhindert fehlerhafte Merges.</td>
      <td><strong>Qualitätssicherung:</strong> Reduziert manuelle Review-Zeiten und verhindert, dass Bugs die Produktion erreichen ("Fail Fast").</td>
    </tr>
    <tr>
      <td><strong><code>azure/infrastructure.tf</code></strong></td>
      <td>Terraform (IaC)</td>
      <td>Infrastructure</td>
      <td>Definiert Cloud-Ressourcen (Compute, Storage) als Code ("Infrastructure as Code").</td>
      <td><strong>Reproduzierbarkeit:</strong> Garantiert, dass die DEV-Umgebung mathematisch identisch zur PROD-Umgebung ist (Verhindert "Works on my machine").</td>
    </tr>
    <tr>
      <td><strong><code>azure/score.py</code></strong></td>
      <td>Entry Script</td>
      <td>Deployment</td>
      <td>Die Brücke zwischen Azure Managed Endpoints und dem Python-Code. Lädt das Modell beim Start.</td>
      <td><strong>Standardisierung:</strong> Notwendig für Azure-native Deployments. Ermöglicht nahtloses Skalieren ohne Code-Änderung.</td>
    </tr>
    <tr>
      <td><strong><code>azure/submit_training_job.py</code></strong></td>
      <td>Training SDK Script</td>
      <td>MLOps</td>
      <td>Sendet rechenintensive Jobs vom lokalen Laptop an einen Azure GPU-Cluster.</td>
      <td><strong>Kosteneffizienz:</strong> Nutzt teure GPUs nur für die Dauer des Trainings (Pay-per-Use) statt dauerhaft laufender Server.</td>
    </tr>
    <tr>
      <td><strong><code>configs/*.env</code></strong></td>
      <td>Environment Variables</td>
      <td>Config</td>
      <td>Trennt Konfiguration (URLs, Debug-Level) von Geheimnissen (API Keys).</td>
      <td><strong>Sicherheit:</strong> Verhindert Hardcoding von Credentials im Code. Ermöglicht einfaches Umschalten zwischen Dev/Prod.</td>
    </tr>
    <tr>
      <td><strong><code>data/raw/vehicle_feedback.csv</code></strong></td>
      <td>Trainingsdaten</td>
      <td>Data Input</td>
      <td>Unveränderte Rohdaten (Kundenfeedback) für das Training des DistilBERT-Modells.</td>
      <td><strong>Datenintegrität:</strong> Die Trennung in <code>raw</code> stellt sicher, dass die Originalquelle niemals versehentlich überschrieben wird.</td>
    </tr>
    <tr>
      <td><strong><code>data/raw/insurance_terms.pdf</code></strong></td>
      <td>Knowledge Base</td>
      <td>Data Input</td>
      <td>Das "Ground Truth" Dokument (Versicherungsbedingungen) für die RAG-Engine.</td>
      <td><strong>Erklärbarkeit:</strong> Dient als faktische Basis für Antworten, um Halluzinationen des LLMs zu verhindern.</td>
    </tr>
    <tr>
      <td><strong><code>data/processed/vector_index.faiss</code></strong></td>
      <td>Vektor-Index</td>
      <td>Artifact</td>
      <td>Der mathematische Index der PDF. Wird offline berechnet und zur Laufzeit nur geladen.</td>
      <td><strong>Latenzreduktion:</strong> Senkt die Startzeit des Containers von Minuten auf Millisekunden und eliminiert redundante Berechnungskosten.</td>
    </tr>
    <tr>
      <td><strong><code>docs/architecture.png</code></strong></td>
      <td>System-Diagramm</td>
      <td>Doku</td>
      <td>Visuelle Darstellung des "Compound AI Systems" (Router + RAG + Security).</td>
      <td><strong>Onboarding:</strong> Beschleunigt das Verständnis komplexer Zusammenhänge für Stakeholder und neue Entwickler.</td>
    </tr>
    <tr>
      <td><strong><code>models/huk_distilbert.onnx</code></strong></td>
      <td>KI-Modell</td>
      <td>Artifact</td>
      <td>Das trainierte, quantisierte Modell im ONNX-Format (Optimiert für CPUs).</td>
      <td><strong>Performance:</strong> ONNX läuft auf Standard-CPUs 4x schneller als PyTorch, was Cloud-Kosten massiv senkt.</td>
    </tr>
    <tr>
      <td><strong><code>notebooks/01_eda.ipynb</code></strong></td>
      <td>EDA Analysis</td>
      <td>Research</td>
      <td>Sandbox zur Analyse von Datenqualität und Klassenverteilung vor dem Engineering.</td>
      <td><strong>Risikominimierung:</strong> Identifiziert "Dirty Data" frühzeitig und verhindert "Garbage In, Garbage Out".</td>
    </tr>
    <tr>
      <td><strong><code>notebooks/02_rag_prototyping.ipynb</code></strong></td>
      <td>RAG PoC</td>
      <td>Research</td>
      <td>Experimentierfeld zum Testen von Chunking-Strategien für die Vektorsuche.</td>
      <td><strong>Entwicklungsgeschwindigkeit:</strong> Erlaubt schnelles Iterieren der Suchlogik, ohne die gesamte App neu starten zu müssen.</td>
    </tr>
    <tr>
      <td><strong><code>reports/figures/confusion_matrix.png</code></strong></td>
      <td>Audit Report</td>
      <td>Reporting</td>
      <td>Visualisierung der Klassifizierungsfehler des Modells.</td>
      <td><strong>Vertrauen:</strong> Dient als technischer Beweis ("Audit Trail") für die Genauigkeit des Modells gegenüber dem Fachbereich.</td>
    </tr>
    <tr>
      <td><strong><code>scripts/ingest_data.py</code></strong></td>
      <td>ETL Script</td>
      <td>DevOps</td>
      <td>Verarbeitet die PDF zu Embeddings *während des Builds*, nicht zur Laufzeit.</td>
      <td><strong>Skalierbarkeit:</strong> Ermöglicht schnelles Auto-Scaling in Azure, da der Index nicht bei jedem Container-Start neu berechnet wird.</td>
    </tr>
    <tr>
      <td><strong><code>src/main_api.py</code></strong></td>
      <td>API Gateway</td>
      <td>Interface</td>
      <td>FastAPI Entry-Point mit Health-Checks (<code>/health</code>) und Request-Tracing.</td>
      <td><strong>Resilienz:</strong> Health-Checks sind essenziell für Azure Load Balancer, um abgestürzte Container automatisch neu zu starten.</td>
    </tr>
    <tr>
      <td><strong><code>src/schemas.py</code></strong></td>
      <td>Pydantic Models</td>
      <td>Security</td>
      <td>Definiert strenge Datenverträge für Input/Output. Validiert User-Eingaben.</td>
      <td><strong>Sicherheit:</strong> Verhindert "Injection Attacks" und Abstürze durch malformierte Daten (Defensive Programming).</td>
    </tr>
    <tr>
      <td><strong><code>src/utils.py</code></strong></td>
      <td>Utilities</td>
      <td>Observability</td>
      <td>Zentralisiertes Logging (JSON), Config-Loading und Performance-Timing.</td>
      <td><strong>Wartbarkeit:</strong> Ein zentraler Ort für Logging-Standards erleichtert die Integration in Azure Monitor / Splunk.</td>
    </tr>
    <tr>
      <td><strong><code>src/exceptions.py</code></strong></td>
      <td>Custom Errors</td>
      <td>Stability</td>
      <td>Definiert eigene Fehlerklassen (z.B. <code>PolicyNotFound</code>) statt generischer Crashes.</td>
      <td><strong>Monitoring:</strong> Erlaubt präzises Alerting (Unterscheidung zwischen User-Fehler 400 und System-Fehler 500).</td>
    </tr>
    <tr>
      <td><strong><code>src/classifier/train.py</code></strong></td>
      <td>Training Loop</td>
      <td>ML Logic</td>
      <td>Logik für Fine-Tuning, Feature Engineering (Tokenization) und Training.</td>
      <td><strong>Reproduzierbarkeit:</strong> Der Code (das "Rezept") ist wichtiger als das Modell, da er den Prozess nachvollziehbar macht.</td>
    </tr>
    <tr>
      <td><strong><code>src/classifier/evaluate.py</code></strong></td>
      <td>Validierung</td>
      <td>ML Logic</td>
      <td>Berechnet Metriken (F1-Score) auf ungesehenen Testdaten.</td>
      <td><strong>Qualität:</strong> Stellt sicher, dass das Modell generalisiert und nicht nur auswendig lernt (Overfitting-Check).</td>
    </tr>
    <tr>
      <td><strong><code>src/classifier/export_onnx.py</code></strong></td>
      <td>Optimierung</td>
      <td>ML Logic</td>
      <td>Konvertiert das schwere PyTorch-Modell in das leichte ONNX-Format.</td>
      <td><strong>Cost Optimization:</strong> Kleinerer Speicherbedarf = Günstigere Cloud-Instanzen.</td>
    </tr>
    <tr>
      <td><strong><code>src/classifier/inference.py</code></strong></td>
      <td>Runtime</td>
      <td>ML Logic</td>
      <td>Lädt das ONNX-Modell und führt die Klassifizierung durch.</td>
      <td><strong>Kapselung:</strong> Trennt die Vorhersage-Logik von der API, was Unit-Testing vereinfacht.</td>
    </tr>
    <tr>
      <td><strong><code>src/rag/cache.py</code></strong></td>
      <td>Semantic Cache</td>
      <td>Scalability</td>
      <td>Speichert Antworten auf häufige Fragen, um OpenAI-Calls zu sparen.</td>
      <td><strong>OpEx Reduktion:</strong> Kann OpenAI-Kosten um 30-50% senken und liefert Antworten in Echtzeit (&lt;10ms).</td>
    </tr>
    <tr>
      <td><strong><code>src/rag/engine.py</code></strong></td>
      <td>Orchestrator</td>
      <td>AI Logic</td>
      <td>Verbindet die Vektorsuche mit dem LLM (Prompt Engineering).</td>
      <td><strong>Kern-Logik:</strong> Hier entsteht die eigentliche "Intelligenz" durch Kombination von Kontext und Sprachmodell.</td>
    </tr>
    <tr>
      <td><strong><code>src/rag/vector_store.py</code></strong></td>
      <td>Retrieval</td>
      <td>AI Logic</td>
      <td>Handhabt die Suche im FAISS-Index nach relevanten PDF-Abschnitten.</td>
      <td><strong>Präzision:</strong> Die Qualität der Suche entscheidet darüber, ob das LLM die richtige Antwort kennt (Recall).</td>
    </tr>
    <tr>
      <td><strong><code>src/security/auth.py</code></strong></td>
      <td>Auth Middleware</td>
      <td>Security</td>
      <td>Prüft API-Keys im Header jeder Anfrage.</td>
      <td><strong>Zugriffskontrolle:</strong> Verhindert unbefugte Nutzung der teuren LLM-Ressourcen (Zero Trust Ansatz).</td>
    </tr>
    <tr>
      <td><strong><code>src/security/pii_scrubber.py</code></strong></td>
      <td>Privacy Firewall</td>
      <td>Security</td>
      <td>Entfernt Namen/IBANs via Regex, *bevor* Daten an OpenAI gehen.</td>
      <td><strong>DSGVO-Compliance:</strong> Essentiell für Versicherungen, um Datenabfluss in die USA zu verhindern.</td>
    </tr>
    <tr>
      <td><strong><code>streamlit_app/app.py</code></strong></td>
      <td>Frontend UI</td>
      <td>Demo</td>
      <td>Chatbot-Interface zur Demonstration des Systems für Stakeholder.</td>
      <td><strong>Sichtbarkeit:</strong> Macht komplexe Backend-Logik für Nicht-Techniker greifbar und testbar.</td>
    </tr>
    <tr>
      <td><strong><code>tests/conftest.py</code></strong></td>
      <td>Test Fixtures</td>
      <td>QA</td>
      <td>Mocks für Azure/OpenAI (Simulierte Antworten für Tests).</td>
      <td><strong>Kostenersparnis:</strong> Verhindert, dass CI/CD-Pipelines echte (kostenpflichtige) API-Calls auslösen.</td>
    </tr>
    <tr>
      <td><strong><code>Dockerfile</code></strong></td>
      <td>Container Config</td>
      <td>Deployment</td>
      <td>Definiert die Laufzeitumgebung (OS, Libraries) für die Cloud.</td>
      <td><strong>Portabilität:</strong> "Write once, run anywhere." Beseitigt Abhängigkeitsprobleme zwischen Dev und Prod.</td>
    </tr>
    <tr>
      <td><strong><code>Makefile</code></strong></td>
      <td>Automation</td>
      <td>Developer Exp</td>
      <td>Kurzbefehle für komplexe Prozesse (<code>make train</code>, <code>make deploy</code>).</td>
      <td><strong>Effizienz:</strong> Beschleunigt das Onboarding neuer Entwickler ("One-Command-Setup").</td>
    </tr>
    <tr>
      <td><strong><code>pyproject.toml</code></strong></td>
      <td>Modern Config</td>
      <td>Tooling</td>
      <td>Zentralisiert Tools (Ruff, Pytest) und ersetzt veraltete <code>setup.py</code>.</td>
      <td><strong>Standard:</strong> Entspricht dem Python-Standard 2025 für sauberes Paketmanagement.</td>
    </tr>
    <tr>
      <td><strong><code>requirements.txt</code></strong></td>
      <td>Dependencies</td>
      <td>Deployment</td>
      <td>Fixierte Versionen aller Bibliotheken ("Pinning").</td>
      <td><strong>Stabilität:</strong> Verhindert, dass Updates von Libraries die Produktion unbemerkt brechen.</td>
    </tr>
  </tbody>
</table>

```bash
This document provides a comprehensive overview of the vehicle feedback sentiment analysis project, covering data loading, exploration, cleaning, model training, evaluation, and deployment to ONNX.
## 1. Data Loading and Initial Exploration
The project began by loading the `vehicle_feedback.csv` dataset.


### Initial Data Snapshot:
- The dataset contains columns such as `category (text)`, `category (binary)`, `sentiment (text)`, `sentiment (binary)`, and `feedback`.
- The `feedback` column contains the raw text data for sentiment analysis.


### Class Balance Check: To understand the distribution of sentiments and categories, the `value_counts()` method was applied, sentiment (text) negative 2581 positive 2549 neutral 2455
### Category (text) balance: category (text) policy 2599 service 2528 claim 2458
The sentiment and category classes appear relatively balanced, which is good for model training as it minimizes bias towards any particular class.


### Inspection for "Dirty" Text and Text Lengths:
- **Unusual Characters:** A regex pattern `r'[^\w\s.,!?@#$%&*()\-+=:;\'\"/\\]'` was used to identify unusual characters in the `feedback` column. This check revealed 51 rows containing such characters, indicating a need for cleaning.
- **Extremely Short Texts:** No extremely short feedback texts (less than 10 characters) were found, which suggests that all feedback entries are substantial enough for analysis.


### Data Types and Basic Statistics:
- **Data Types:** All `(text)` columns (`category (text)`, `sentiment (text)`, `feedback`, `feedback_clean`) were of `object` type, while `(binary)` columns (`category (binary)`, `sentiment (binary)`) were `int64`. The `feedback_clean` column was added later as an `object`.

- **Basic Statistics:** The `describe(include='all')` function provided summaries. For categorical columns, it showed counts, unique values, and the top occurring item. For numerical columns, it provided standard descriptive statistics.


## 2. Data Cleaning Steps: The data cleaning process focused on standardizing the text data and handling structural issues.
### Cleaning Unusual Characters:
- A corrected regular expression `clean_pattern = r"[^a-zA-Z0-9\s.,!?'\"-]"` was applied to the `feedback` column to remove characters that are not alphanumeric, whitespace, or common punctuation. This resulted in a new `feedback_clean` column.
- Example of cleaning:
    - Original: `# Inefficient communication and lack of transp...`
    - Cleaned: ` Inefficient communication and lack of transpa...` (the '#' was removed).


### Handling Duplicates and Missing Values:
- **Duplicates:** 110 duplicate `feedback` entries were found and subsequently entfernt, ensuring that the model is not overfitted to redundant information.
- **Missing Values:** A check on important columns (`feedback`, `sentiment (text)`, `category (text)`) confirmed no missing values, eliminating the need for imputation or removal of rows due to nulls.






## 3. Data Visualization
Visualizations were used to understand text characteristics and class distributions.
### Distribution of Feedback Text Lengths:
- A new column `feedback_length` was created from `feedback_clean`.
- A histogram revealed that feedback lengths generally ranged from about 40 to 3700 characters, with a mean of 433.9 and a standard deviation of 242.9.
- Outlier detection (outside the 1st-99th percentile) identified 147 unusually long or short feedbacks, which could be investigated further if necessary for specific use cases.


### Feedback Counts by Sentiment and Category: 
- A bar plot illustrated the distribution of sentiment (negative, neutral, positive) within each category (claim, policy, service).
- It highlighted key observations, such as 834 negative feedbacks specifically related to 'claim' category, indicating potential areas for improvement in claims processing.


## 4. Model Training (DistilBERT for Sentiment Classification)
A DistilBERT model was fine-tuned for sentiment classification.


### Data Preparation:
- The dataset was split into training and testing sets (80/20 ratio) using `X_train`, `X_test`, `y_train`, `y_test`. `X` was the `feedback_clean` text, and `y` was `sentiment (text)`. Stratification was used to maintain class proportions.


 - Train size: 6068, Test size: 1517


- `DistilBertTokenizerFast` from `distilbert-base-uncased` was used to tokenize the text data.
    - `train_encodings` and `test_encodings` were created with `truncation=True`, `padding=True`, and `max_length=256`.


- Labels were mapped to integers using `label2id = {'negative': 0, 'neutral': 1, 'positive': 2}`.
- A custom `FeedbackDataset` class was created to handle the tokenized inputs and integer labels.


### Model and Training Configuration:
- `DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label2id))` was loaded.
- `TrainingArguments` were set up:
    - `output_dir='./results'`
    - `num_train_epochs=2`
    - `per_device_train_batch_size=16`
    - `per_device_eval_batch_size=16`
    - `eval_strategy='epoch'`
    - `save_strategy='epoch'`
    - `logging_dir='./logs'`
    - `logging_steps=50`
    - `load_best_model_at_end=True`
    - `metric_for_best_model='eval_loss'`
    - `save_total_limit=2`
    - `fp16=False` (disabled for CPU compatibility)
    - `report_to='none'`
    - `seed=42`
    - `optim='adamw_torch'`




### Training Output: 
The training concluded successfully, as indicated by the `TrainOutput`:
Weighted F1-score: 0.9861505113816377
Confusion Matrix:
 [[513   3   0]
 [  3 480   8]
 [  1   6 503]]
513 negative feedbacks were correctly classified as negative.
480 neutral feedbacks were correctly classified as neutral.
503 positive feedbacks were correctly classified as positive.
This report indicates excellent performance across all sentiment categories:
Precision (0.98-0.99): The model is very good at identifying only relevant instances for each class.
Recall (0.98-0.99): The model is very good at finding all relevant instances for each class.
F1-score (0.98-0.99): This is the harmonic mean of precision and recall, showing a strong balance between the two metrics for all classes.
Accuracy (0.98): The overall accuracy of the model is very high.
These metrics collectively demonstrate that the fine-tuned DistilBERT model is performing exceptionally well in classifying the sentiment of the feedback.
              precision    recall  f1-score   support

    negative       0.99      0.99      0.99       516
     neutral       0.98      0.98      0.98       491
    positive       0.98      0.99      0.99       510

    accuracy                           0.99      1517
   macro avg       0.99      0.99      0.99      1517
weighted avg       0.99      0.99      0.99      1517




## 5. Model Evaluation


The trained model's performance was evaluated using F1-score, confusion matrix, and a classification report.


### Evaluation Metrics:
    - A very high F1-score indicates excellent balance between precision and recall across all classes, reflecting strong overall model performance.


- **Confusion Matrix:**
    - The confusion matrix shows very few misclassifications, with the diagonal elements being significantly larger than off-diagonal elements, confirming high accuracy for each class.
    - The classification report corroborates the F1-score, with precision, recall, and F1-score all at or near 0.99 for all classes. This indicates that the model is performing exceptionally well in classifying sentiments.


## ONNX Model Export & Integration (30. November 2025)

I exported the fine-tuned DistilBERT model to ONNX format for efficient deployment and inference. The ONNX model is located at `models/huk_distilbert.onnx`.

### ONNX Model Details
- The ONNX model is optimized for CPU inference and is used in both local and cloud deployments.
- Inference is performed using `onnxruntime` in the scoring script (`azure/score.py`).
- The tokenizer configuration is loaded from `configs/tokenizer_files/` to ensure input compatibility.

### ONNX Integration Steps
1. Load the ONNX model with `onnxruntime.InferenceSession`.
2. Tokenize input text using `DistilBertTokenizerFast`.
3. Prepare `input_ids` and `attention_mask` as NumPy arrays for ONNX inference.
4. Run inference and map the output to sentiment labels using the label map.

### Example ONNX Inference (Python)
```python
import onnxruntime as ort
from transformers import DistilBertTokenizerFast
import numpy as np

session = ort.InferenceSession('models/huk_distilbert.onnx')
tokenizer = DistilBertTokenizerFast.from_pretrained('configs/tokenizer_files/')
text = "The insurance claim process was fast and easy."
inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='np')
ort_inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
logits = session.run(None, ort_inputs)[0]
pred_id = int(np.argmax(logits, axis=1)[0])
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
pred_label = label_map[pred_id]
print({'prediction': pred_label, 'score': float(np.max(logits))})
```

### Integration Points
- The ONNX model is used in `aws/score.py` for Azure ML managed endpoint deployment.
- The same inference logic is used in local and Docker deployments for consistency.

---

## RAG Workflow Overview

1. **PDF Ingestion**: Extract text from German PDF files (e.g., `data/raw/insurance_terms.pdf`).
2. **Translation**: Translate German text to English using Azure OpenAI or other translation service.
3. **Embedding**: Encode translated text into vector embeddings using DistilBERT or similar model.
4. **Vector Indexing**: Store embeddings in FAISS index for efficient similarity search.
5. **Query & Retrieval**: Accept user queries, embed them, and perform vector search to retrieve relevant document chunks.

---

## Example RAG API Usage

### Ingest Data
```bash
python scripts/ingest_data.py --pdf_path data/raw/insurance_terms.pdf --index_path data/processed/faiss_index.bin
```

### Query Endpoint (FastAPI)
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the process for vehicle insurance claims?"}'
```
**Response:**
```json
{
  "results": [
    {
      "chunk": "The insurance claim process involves...",
      "score": 0.92
    }
  ]
}
```

---

## Testing Instructions

### Local Testing
- Run the FastAPI app locally:
  ```bash
  uvicorn FastAPI_app.app:app --reload --env-file configs/dev.env
  ```
- Test RAG endpoints with curl or Postman as shown above.

### Docker Testing
- Build and run Docker image:
  ```bash
  docker build -t huk-feedback-app .
  docker run --env-file configs/dev.env -p 8000:8000 huk-feedback-app
  ```
- Test endpoints at `http://localhost:8000/rag/query`

### Azure Testing
- Deploy using Azure ML or App Service (see deployment doc).
- Ensure environment variables are set from `prod.env`.
- Test the RAG engine via the deployed API endpoint.

---

I have updated this section to clearly describe the ONNX model export, integration, and usage in your project pipeline.

## Local ONNX Model Inference Test (30. November 2025)

I successfully ran a local test of the ONNX model inference using `azure/test_score_local.py` on macOS. The scoring script loaded the ONNX model and tokenizer, processed the input, and returned a valid prediction:

```
Inference result: {"prediction": "negative", "score": 0.6555681228637695}
```

This confirms that:
- The ONNX model and tokenizer are correctly integrated in `azure/score.py`.
- The preprocess and run functions work as expected for local inference.
- The output format matches the requirements for downstream integration and deployment.

Next steps: I will proceed to Azure ML managed endpoint deployment or integrate the scoring logic into the FastAPI app for production use.

## ONNX Model Inference with Azure ML

### Step 1: Prepare the Azure ML Scoring Script
I updated `azure/score.py` to:
- Load the ONNX model (`models/huk_distilbert.onnx`) using `onnxruntime`.
- Load the DistilBERT tokenizer from `configs/tokenizer_files/` using HuggingFace `transformers`.
- Preprocess input text with the actual tokenizer, matching the training pipeline.
- Run inference and return predictions with the correct label mapping.

### Step 2: Deploy the Model to Azure ML
I will use this scoring script as the entry point when deploying the ONNX model to an Azure ML managed endpoint. This ensures that inference in the cloud matches local results.

### Step 3: Test and Validate
After deployment, I will test the endpoint by sending sample requests and verifying that predictions are returned correctly. I will update the scoring script as needed to improve preprocessing and error handling.

### Next Steps
- Automate deployment using Azure ML CLI or SDK.
- Document API usage and expected input/output formats.
- Integrate with the FastAPI app or other services as needed.

For details on the scoring script, see `azure/score.py`. For deployment instructions, see the Azure ML documentation and repo scripts.

## Azure OpenAI RAG Integration (30. November 2025)

I integrated Azure OpenAI environment variables into the RAG engine (`src/rag/engine.py`). The engine now loads credentials and endpoint info from `configs/dev.env` and `configs/prod.env` and can query the Azure OpenAI deployment for completions.

### How it works:
- The RAG engine uses the function `query_azure_openai(prompt)` to send requests to the Azure OpenAI endpoint.
- Credentials and endpoint info are loaded from environment variables for security.
- The workflow is ready for end-to-end testing and further integration with FastAPI or other services.

**Next steps:**
- Test the RAG workflow with real queries and vector search.
- Document example usage and API endpoints for RAG in the docs.
- Ensure secrets and sensitive files are excluded from version control (see updated `.gitignore`).

## RAG Vector Index Creation with PDF Translation (30. November 2025)

I updated the ingestion pipeline in `scripts/ingest_data.py` to automatically translate the German PDF (`insurance_terms.pdf`) to English using the `googletrans` library. The translated text is then embedded with DistilBERT and indexed using FAISS for semantic search.

### Steps:
1. Extract text from the German PDF using PyPDF2.
2. Translate each page to English with Google Translate (via `googletrans`).
3. Generate embeddings for the translated text using DistilBERT.
4. Build and save a FAISS vector index to `data/processed/vector_index.faiss`.

This ensures the RAG engine can work with English queries and context, even if the source document is in German.

**Next steps:**
- Test the RAG workflow locally, in Docker, and on Azure.
- Document example queries and API endpoints for the RAG engine.
- Continue integration with FastAPI and update documentation as needed.

## RAG Workflow Checklist & Example Usage (30. November 2025)

### RAG Engine Information
- I have set up the Azure OpenAI API key and endpoint in `configs/dev.env` and `configs/prod.env`.
- The vector index (`data/processed/vector_index.faiss`) is built from the German PDF (`insurance_terms.pdf`), automatically translated to English before embedding.
- The RAG engine uses Azure OpenAI endpoints for completions.

### RAG Documentation & Example Queries
- The ingestion pipeline extracts, translates, embeds, and indexes PDF content for semantic search.
- Example query for the RAG engine:

```python
from src.rag.engine import query_azure_openai
response = query_azure_openai("What is covered under the insurance policy?")
print(response)
```
- API endpoint (if integrated with FastAPI):
```
POST /rag/query
{
  "query": "What is covered under the insurance policy?"
}
```

### Configuration
- All secrets and endpoints are managed in `.env` files and excluded from version control.
- The RAG engine can be tested locally, in Docker, or deployed to Azure.

### Next Steps
- Test the RAG workflow end-to-end in all environments.
- Expand documentation with more example queries and integration tests as needed.
- Integrate the RAG engine with the FastAPI app for production use.

## RAG Flow Documentation (30. November 2025)

### RAG Pipeline Steps
1. **PDF Extraction:**
   - I extract text from each page of the German PDF (`insurance_terms.pdf`) using PyPDF2.
2. **Translation:**
   - I automatically translate the extracted German text to English using the `googletrans` library.
3. **Embedding:**
   - I generate embeddings for the translated English text using DistilBERT.
4. **Vector Indexing:**
   - I build a FAISS vector index from the embeddings and save it to `data/processed/vector_index.faiss`.

### Example RAG Query (Python)
```python
from src.rag.engine import query_azure_openai
response = query_azure_openai("What is covered under the insurance policy?")
print(response)
```

### Example API Endpoint Usage (FastAPI)
```
POST /rag/query
{
  "query": "What is covered under the insurance policy?"
}
```

### Testing Instructions
#### Local
- Run the ingestion script to build the vector index:
  ```bash
  python3 scripts/ingest_data.py
  ```
- Test RAG queries using Python or FastAPI locally.

#### Docker
- Build the Docker image:
  ```bash
  docker build -t huk-feedback-app .
  ```
- Run the container with environment variables:
  ```bash
  docker run --env-file configs/dev.env -p 8000:8000 huk-feedback-app
  ```
- Send API requests to `http://localhost:8000/rag/query`.

#### Azure
- Deploy the Docker image or FastAPI app to Azure (e.g., Azure App Service or Azure ML endpoint).
- Ensure environment variables are set from `prod.env`.
- Test the RAG engine via the deployed API endpoint.

---
I have documented the full RAG flow, provided example queries and API usage, and included instructions for local, Docker, and Azure testing.

## Azure ML Deployment & Testing Instructions (30. November 2025)

### Infrastructure as Code (Terraform)
- Use `azure/infrastructure.tf` to provision Azure resources for ML deployment:
  1. Resource group
  2. Container registry
  3. ML workspace
  4. Managed online endpoint
- Run these commands to deploy:
  ```bash
  az login
  terraform init
  terraform apply
  ```

### Model & API Deployment
- Build and push your Docker image to Azure Container Registry (ACR).
- Deploy the image to the managed endpoint using Azure ML Studio or CLI.
- Use `azure/score.py` as the entry script for ONNX inference.

### Testing
- After deployment, test the endpoint with sample requests:
  ```bash
  curl -X POST https://<endpoint-url>/score \
    -H "Content-Type: application/json" \
    -d '{"text": "The insurance claim process was fast and easy."}'
  ```
- Check logs and responses for successful inference.

### Local & Docker Testing
- Build and run locally:
  ```bash
  docker build -t huk-feedback-app .
  docker run --env-file configs/dev.env -p 8000:8000 huk-feedback-app
  ```
- Test API endpoints at `http://localhost:8000`.

👨‍💻 Autor
Hassan Daoud