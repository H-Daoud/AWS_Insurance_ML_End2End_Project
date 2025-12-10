# Data and Model Development
⚡ **Schlüsselkomponenten**
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
- **Duplicates:** 110 duplicate `feedback` entries were found and subsequently removed, ensuring that the model is not overfitted to redundant information.
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


## 6. Model Export to ONNX
The trained PyTorch model was exported to ONNX format for efficient deployment and inference.
- The best performing model from training was saved to `./models/distilbert_huk_sentiment`.
- `torch.onnx.export` was used to convert the model.


## 7. ONNX Model Testing
The exported ONNX model was tested to ensure correct inference behavior.
### Setup:
- `onnxruntime` was installed.
- The ONNX model `huk_distilbert.onnx` was loaded using `ort.InferenceSession` with `CPUExecutionProvider`.
- `DistilBertTokenizerFast` was reloaded.
- The `id2label` mapping (`{0: 'negative', 1: 'neutral', 2: 'positive'}`) was used to interpret predictions.


### `predict_sentiment` Function:
A function `predict_sentiment(text)` was created to:
1. Tokenize input text using the `DistilBertTokenizerFast`.
2. Prepare inputs (`input_ids`, `attention_mask`) for the ONNX session as NumPy arrays.
3. Run inference using `session.run()`.
4. Apply softmax to the raw logits to obtain probabilities.
5. Determine the predicted label ID using `np.argmax`.
6. Map the ID back to the sentiment label using `id2label`.


### Sample Feedback Predictions:
- **Sample 1 (Positive):**
    - Feedback: "I am extremely satisfied with the excellent service provided. The staff was very helpful and resolved my issue quickly."
    - Predicted Sentiment (ONNX): `neutral`
    - Probabilities: `[0.43987906 0.48040444 0.07971653]`
    - **Note:** The model incorrectly predicts 'neutral' here, with 'neutral' having the highest probability, followed by 'negative', and 'positive' being the lowest. This suggests a potential issue in the ONNX export or the `predict_sentiment` function's interpretation, as the original PyTorch model achieved high accuracy. The low probability for 'positive' indicates that the model's output distribution might be different or that the specific sample is ambiguous.


- **Sample 2 (Negative):**
    - Feedback: "I'm very disappointed with the long waiting times and unresponsive customer support."
    - Predicted Sentiment (ONNX): `neutral`
    - Probabilities: `[0.4373155  0.48306382 0.0796207 ]`
    - **Note:** Similar to the positive sample, the model incorrectly predicts 'neutral' for a clearly negative feedback, with 'neutral' probability being highest.


- **Sample 3 (Neutral):**
    - Feedback: "The policy terms are standard and the process was neither good nor bad."
    - Predicted Sentiment (ONNX): `neutral`
    - Probabilities: `[0.43960977 0.48066732 0.07972285]`
    - **Note:** This neutral sample is correctly predicted as 'neutral', with the highest probability.


 - **Correction and Observation:** The consistent prediction of 'neutral' with probabilities around 0.48 for the `neutral` class (which is index 1 in `id2label` with the current mapping) and similar low probabilities for 'negative' (index 0) and 'positive' (index 2) across various test samples, even those with clear positive or negative sentiment, indicates an issue. The model's logits are not clearly separating classes, and the softmax probabilities are nearly uniform (around 0.33 each) but slightly favoring 'neutral' due to small numerical differences. This suggests that the ONNX model is not performing as expected compared to the high F1-score achieved during evaluation with the PyTorch model. **Further investigation is required to align the ONNX model's output behavior with the fine-tuned PyTorch model.** This could involve re-checking the ONNX export process, the `id2label` mapping, or the `predict_sentiment` function. However, based on the metrics reported, the PyTorch model was performing very well.
#### Model Conclusion: The project successfully established a pipeline for sentiment analysis of vehicle feedback, from data preparation to model training and ONNX export. While the PyTorch model demonstrated high performance in evaluation metrics, the ONNX model's performance in the final tests needs further debugging to ensure it replicates the expected behavior. The generated `models.zip` and `results.zip` contain the model artifacts and training logs.
Data Analysis Key Findings
Data Quality:
The initial dataset contained 51 rows with unusual characters in the feedback column, which were subsequently cleaned.
110 duplicate feedback entries were identified and removed, ensuring data uniqueness.
No missing values were found in critical columns (feedback, sentiment (text), category (text)).
Data Characteristics:
Sentiment classes (negative, positive, neutral) and category classes (policy, service, claim) were relatively balanced, with counts ranging from 2455 to 2599 for text sentiments and categories, respectively.
Feedback text lengths varied from 42 to 3783 characters, with an average length of 433.9 characters.
Model Performance (PyTorch):
The DistilBERT sentiment classification model achieved excellent performance with a weighted F1-score of 0.986.
Precision, recall, and F1-scores were consistently around 0.99 for all sentiment classes (negative, neutral, positive), indicating high accuracy and minimal misclassifications.
ONNX Export and Inference Discrepancy:
The PyTorch model was successfully exported to the ONNX format.
However, testing of the ONNX model revealed a significant issue: it consistently predicted 'neutral' for various inputs, even for clearly positive or negative feedback. The probabilities for 'neutral' were around 0.48, while 'negative' and 'positive' probabilities were much lower, suggesting a problem in the ONNX model's inference behavior compared to the highly accurate PyTorch model.
Insights / Next Steps
Investigate ONNX Model Discrepancy: The primary next step is to thoroughly investigate why the ONNX model is not replicating the high performance of the original PyTorch model. This could involve re-evaluating the ONNX export parameters, the input processing for ONNX inference, or the id2label mapping within the predict_sentiment function to ensure consistency.
Deployment Readiness: Until the ONNX model's inference behavior aligns with the PyTorch model's evaluated performance, it is not suitable for deployment. Rectifying this issue is crucial for efficient and accurate sentiment analysis in production.

👨‍💻 Autor
Hassan Daoud