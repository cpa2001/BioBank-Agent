# UK Biobank Agent: Comprehensive Analysis of Foundation Models & Analysis Approaches

**Research Exploration Report**  
**Generated:** 2026-04-23  
**Project:** `/Users/chenpengan/Projects/CUHK/UKB_agent/`

---

## Executive Summary

The Biobank Agent is an **LLM-powered autonomous scientific discovery system** for population-scale biobank research. It combines a **ReAct agent loop** with **45+ specialized analysis skills** to enable end-to-end research workflows. 

### Key Finding: Foundation Models Architecture

The codebase includes **abstract interfaces for foundation models that are NOT YET DEPLOYED**:
- `biobank_agent/interfaces/fm_embedding.py` - Abstract interface for remote FM servers
- `biobank_agent/interfaces/multimodal.py` - Multimodal data fusion framework
- Planned models: **Evo2** (genomics), **ESM-2** (proteomics), **BrainLM** (imaging)

**Current Status:** Built-in skills use **traditional ML** (XGBoost, LightGBM, CatBoost) with **scikit-learn embeddings** (t-SNE, UMAP). Foundation models are architectural placeholders waiting for server deployment.

---

## 1. FOUNDATION MODELS & EMBEDDINGS INFRASTRUCTURE

### 1.1 Abstract Foundation Model Interface

**File:** `biobank_agent/interfaces/fm_embedding.py`

```python
class FMEmbeddingInterface(ABC):
    """Abstract base for calling remote foundation models."""
    
    def encode(self, data: pd.DataFrame, modality: str, model_name=None) -> np.ndarray:
        """Extract embeddings from data using a remote FM.
        
        Parameters:
        - data: DataFrame (rows = samples)
        - modality: 'genomic', 'protein', 'imaging', etc.
        - model_name: optional FM override
        
        Returns: np.ndarray (n_samples, embedding_dim)
        """
```

### 1.2 Planned Foundation Models (Awaiting Server Deployment)

| Model | Modality | Embedding Dim | Status | Purpose |
|-------|----------|---------------|--------|---------|
| **Evo2** | Genomic | 1024 | ❌ Not deployed | DNA sequence encoding |
| **ESM-2** | Protein | 1280 | ❌ Not deployed | Protein language model |
| **BrainLM** | Imaging | 768 | ❌ Not deployed | Brain MRI encoding |

**Configuration:** Activate via `FM_SERVER_URL` in `.env` (currently raises `NotImplementedError`)

### 1.3 Multimodal Fusion Interface

**File:** `biobank_agent/interfaces/multimodal.py`

```python
class SimpleMultimodalGrounder:
    """MVP multimodal fusion with graceful degradation."""
    
    def fuse(self, modalities: dict[str, np.ndarray], method: str = "concat") -> np.ndarray:
        """Combine embeddings from multiple modalities."""
        # Methods: 'concat' (default), 'zscore_concat'
    
    def extract_structural_signals(self) -> List[StructuralSignal]:
        """Lightweight anomaly detection from available modalities."""
        # Signals: missingness_ratio, outlier_ratio, variance_proxy
    
    def figure_to_tensor(self, figure_path: str) -> np.ndarray:
        """Convert plots to tensor representation (128×128 grayscale)."""
```

**Key Design:** Degrades gracefully when Pillow unavailable (fallback to filesystem metadata embedding).

---

## 2. CURRENT ANALYSIS CAPABILITIES (45 Built-In Skills)

### 2.1 Predictive Modeling Skills

| Skill | ML Engine | Features |
|-------|-----------|----------|
| **train_model** | XGBoost / LightGBM / CatBoost | 5-fold CV, AUC + 95% CI, stores best model in session state |
| **predict** | Binary classifier (predict_proba) | Risk scores, risk categories (low/medium/high), holds-out test set |
| **evaluate_model** | Cross-validation metrics | AUC, F1, precision, recall, confusion matrix |
| **feature_importance** | Tree-based or SHAP | Top-N feature ranking with horizontal bar chart (Nature style) |
| **calibration** | Reliability diagram | ECE, MCE, Brier score, calibration curves |

### 2.2 Association & Discovery Skills

| Skill | Method | Output |
|-------|--------|--------|
| **gwas_proxy** | Mann-Whitney U + FDR/Bonferroni | Phenotype-wide association, Manhattan plot |
| **phewas** | Univariate biomarker-vs-disease | PheWAS Manhattan plot (ICD10 chapters) |
| **biomarker_dist** | Violin plots + Mann-Whitney | Case-vs-control biomarker comparison |
| **correlation** | Pearson + hierarchical clustering | Heatmap with dendrogram |
| **comorbidity** | Odds ratio calculation | Disease co-occurrence network |
| **discover** | Full pipeline orchestration | Cohort → model → features → PheWAS → literature |

### 2.3 Survival & Epidemiology Skills

| Skill | Method | Data |
|-------|--------|------|
| **survival** | Kaplan-Meier + log-rank test | Uses actual death dates, follow-up times from recruitment |
| **prevalence** | Frequency analysis | Top N diseases by ICD10 chapter |
| **cohort_summary** | Cohort statistics | N cases, controls, demographics |
| **missing_data** | Pattern analysis | Missing data heatmap & dropout patterns |

### 2.4 Visualization & Reporting

| Skill | Purpose |
|-------|---------|
| **smart_plot** | Publication-quality figures (Nature/ICML/NEJM/Lancet styles) |
| **nature_writer** | Nature-quality prose generation from analysis results |
| **report** | Markdown/HTML/PDF report compilation |
| **embedding** | t-SNE/UMAP patient embeddings (colored by disease status) |

### 2.5 Research & Literature Skills

| Skill | Purpose |
|-------|---------|
| **web_search** | DuckDuckGo/Brave/Serper search |
| **web_fetch** | Fetch & parse web content |
| **fetch_paper** | arXiv/PubMed paper retrieval |
| **read_paper** | PDF parsing + semantic chunking |
| **deep_research** | Multi-source synthesis (search + papers) |

### 2.6 Self-Evolution Skills

| Skill | Purpose |
|-------|---------|
| **create_skill** | Generate new skills at runtime |
| **record_macro** | Save analysis pipelines as reusable workflows |
| **replay_pipeline** | Re-run saved pipelines |
| **track_error** | Error catalog & pattern analysis |
| **suggest_error_fix** | LLM-powered error debugging |
| **brainstorm** | Hypothesis generation |
| **critical_thinking** | Multi-perspective analysis review |

---

## 3. DATA MODALITIES CURRENTLY SUPPORTED

### 3.1 Available Biomarker Groups

**File:** `biobank_agent/data/features.py`

```python
BLOOD_BIOCHEMISTRY = {
    "30600": "Albumin",
    "30610": "Alkaline phosphatase",
    "30620": "Alanine aminotransferase",
    "30690": "Cholesterol",
    "30700": "Creatinine",
    "30710": "C-reactive protein",  # Inflammation marker
    "30740": "Glucose",
    "30750": "Glycated haemoglobin (HbA1c)",
    "30760": "HDL cholesterol",
    "30780": "LDL direct",
    "30850": "Testosterone",
    "30870": "Triglycerides",
    "30880": "Urate",
    "30890": "Vitamin D",
}

BLOOD_COUNT = {
    "30000": "White blood cell count",
    "30010": "Red blood cell count",
    "30020": "Haemoglobin concentration",
    "30030": "Haematocrit percentage",
    "30080": "Platelet count",
    "30120-30160": "Differential leukocyte counts (lymph, mono, neutro, eos, baso)",
}

ANTHROPOMETRIC = {
    "48": "Waist circumference",
    "49": "Hip circumference",
    "50": "Standing height",
    "21001": "Body mass index (BMI)",
}

BLOOD_PRESSURE = {
    "4079": "Diastolic BP (automated)",
    "4080": "Systolic BP (automated)",
    "102": "Pulse rate",
}
```

### 3.2 Data NOT Currently Integrated (But Mentioned as Architectural Gaps)

❌ **Genomics Data:**
- GWAS results (mentioned in skill names like `gwas_proxy`, but using biomarker associations as proxy)
- SNP data / Polygenic risk scores (PRS)
- Whole exome sequencing (WES)
- Whole genome sequencing (WGS)

❌ **Proteomics:**
- Olink (proximity extension assay)
- SomaScan (aptamer-based)
- No specific field IDs registered

❌ **Metabolomics:**
- NMR spectroscopy data
- Not referenced in feature groups

❌ **Imaging:**
- Brain MRI (T1/T2/fMRI)
- Retinal photography
- DXA scans (bone density)
- Foundation model placeholder (BrainLM) exists but not deployed

❌ **Accelerometer Data:**
- No fields registered
- Not mentioned in code

❌ **ECG Data:**
- No dedicated fields
- Not referenced in skills

❌ **Text/NLP:**
- Clinical notes not processed
- Web search used only for literature, not clinical text

**Note:** These are referenced only in the **FM embedding interface** as future modalities when servers are deployed.

---

## 4. AGENT EXECUTION ARCHITECTURE

### 4.1 ReAct Agent Loop

**File:** `biobank_agent/agent.py`

```
User Query
    ↓
[Build Context: DM + Catalog + State + Memory]
    ↓
[Call LLM with Tool Definitions (45 skills)]
    ↓
[LLM Routes to Appropriate Skill(s)]
    ↓
[Execute Skill via Registry]
    ↓
[Generate Figure + Store Result]
    ↓
[Next Tool Call or Return]
```

### 4.2 Multi-Model Orchestration (Optional)

**File:** `biobank_agent/orchestrator.py`

For **complex tasks** (complexity score > threshold), system can:
1. Route to multiple frontier models (GPT-5.4, Gemini-3.1-pro)
2. Run red/blue debate with anonymous voting
3. Verify outputs with execution-grounded safety checks
4. Aggregate claims with confidence scores

```python
MultiModelOrchestrator(
    model_pool=[ModelSpec("gpt-5.4"), ModelSpec("gemini-3.1-pro")],
    complexity_threshold=0.7,
    debate_rounds=3,
)
```

### 4.3 Skill Registry & Execution

**File:** `biobank_agent/registry.py`

```python
class SkillRegistry:
    def execute(self, name: str, args: dict, ctx) -> dict:
        """
        1. Look up function by name
        2. Import module if lazy-loaded
        3. Inject ctx as keyword argument
        4. Call func(**args, ctx=ctx)
        """
```

**Skills are discovered via:**
- `@skill` decorator introspection at startup
- All 45 built-in skills auto-loaded from `biobank_agent/skills/`
- Custom skills loaded from `CUSTOM_SKILLS_DIR` if configured

---

## 5. PYTHON DEPENDENCIES & VERSIONS

**File:** `pyproject.toml`

### Core Data & ML Stack
```toml
duckdb>=1.0                    # SQL on parquet/CSV (zero-copy)
pyarrow>=15.0                  # Arrow format support
pandas>=2.0                    # Dataframe manipulation
numpy>=1.24                    # Numerical computing
scikit-learn>=1.4              # ML utilities (preprocessing, metrics)
xgboost>=2.0                   # Gradient boosting (primary model)
lightgbm>=4.0                  # Fast GBDT (alternative model)
```

### Optional GPU Dependencies
```toml
catboost>=1.2                  # CatBoost classifier (optional)
shap>=0.44                      # SHAP explanations (optional)
umap-learn>=0.5                # UMAP dimensionality reduction (optional)
```

### Visualization & Reporting
```toml
matplotlib>=3.8                # Publication-quality figures
seaborn>=0.13                  # Statistical plots
```

### Analysis & Statistics
```toml
lifelines>=0.28                # Kaplan-Meier survival analysis
scipy>=1.12                    # Statistical tests (Mann-Whitney, log-rank)
networkx>=3.2                  # Network analysis (comorbidity graphs)
```

### LLM & Agent
```toml
openai>=1.50                   # OpenAI API (Claude, GPT via relay)
rich>=13.0                     # Terminal formatting
prompt-toolkit>=3.0            # CLI interface
```

### Utilities
```toml
pymupdf>=1.24                  # PDF reading (papers)
httpx>=0.27                    # HTTP client
beautifulsoup4>=4.12           # HTML parsing
duckduckgo-search>=6.0         # Web search
python-dotenv>=1.0             # .env configuration
pydantic>=2.0                  # Settings validation
pyyaml>=6.0                    # YAML parsing
```

### Framework Info
- **Python:** ≥ 3.10
- **Architecture:** Tested on macOS ARM + Linux x86 with CUDA fallback
- **Database:** DuckDB (in-memory with parquet views)

---

## 6. DATA MANAGEMENT & SOURCES

### 6.1 Data Input Formats

**Parquet (Fast Path):**
```
UKB_PARQUET_DIR/
  ├── biomarkers/*.parquet       # Blood tests, anthropometrics, vitals
  ├── diagnoses/*.parquet         # ICD10 diagnosis codes
  └── deaths/*.parquet            # Death codes + dates
```

**CSV (Fallback):**
```
UKB_RAW_DIR/
  ├── ukb672073_Population_Characteristics.csv
  ├── ukb672073_Biological_Samples.csv
  ├── ukb672073_Health_Related_Outcomes.csv
  ├── ukb672073_Additional_Exposures.csv
  └── ...
```

### 6.2 Data Catalog

**File:** `biobank_agent/data/catalog.py`

- **11,821 fields** from UK Biobank (application 672073)
- **410 categories** (Population, Biological, Health Outcomes, etc.)
- Field catalogue enables human-readable lookups (field ID → name)
- Configurable for other biobanks (FinnGen, China Kadoorie, etc.)

### 6.3 DataManager Architecture

**File:** `biobank_agent/data/loader.py`

```python
class DataManager:
    """DuckDB-based unified query layer."""
    
    def query(sql: str) -> pd.DataFrame
        # Routes to fastest source (parquet views first, CSV fallback)
    
    def get_field(field_id: str) -> pd.DataFrame
        # Fetch specific biomarker across all subjects
    
    def count_subjects() -> int
        # Total population size
```

---

## 7. SESSION STATE & MEMORY SYSTEM

### 7.1 Short-Term State

**File:** `biobank_agent/state.py`

```python
class SessionState:
    models: dict[str, Estimator]           # Trained ML models
    model_metadata: dict[str, dict]        # Model config + metrics
    cohorts: dict[str, pd.DataFrame]       # Case/control datasets
    feature_matrix: pd.DataFrame           # Current X matrix
    labels: pd.Series                      # Current y vector
    figures: list[str]                     # Generated figure paths
    records: list[AnalysisRecord]          # Execution log
```

### 7.2 Long-Term Memory

**File:** `biobank_agent/memory.py`

```python
class LongTermMemory:
    # Tier 1: Model configs (best AUC by disease/model type)
    # Tier 2: Analysis pipelines (workflow definitions)
    # Tier 3: Error patterns (common failures + fixes)
    # Tier 4: Field usage (popularity of biomarker fields)
```

Persisted to `~/.biobank_agent/memory.json`

---

## 8. REPORT GENERATION

### 8.1 Output Formats

**Markdown → HTML → PDF (via pandoc)**

**Styles Supported:**
- Nature journals (Arial, 89mm single col)
- ICML/NeurIPS (Times, 6.75" textwidth)
- NEJM (86mm single col)
- Lancet (89mm single col)

### 8.2 Report Sections

```
├── Executive Summary
├── Key Findings (highlighted box)
├── Methods
│   ├── Data Source
│   ├── Cohort Definition
│   └── Statistical Methods
├── Results
│   ├── Figure 1: Prevalence
│   ├── Figure 2: Model ROC
│   ├── Figure 3: Feature Importance
│   └── Figure 4: PheWAS Manhattan
└── Discussion + References
```

All figures in **Nature style:**
- SVG + PDF (300 DPI)
- Okabe-Ito color palette (colorblind-friendly)
- No top/right spines
- Arial 7-8pt font

---

## 9. KNOWN ARCHITECTURAL GAPS & FUTURE WORK

### 9.1 Foundation Models (Not Yet Deployed)

- [ ] Evo2 server deployment (genomic embeddings)
- [ ] ESM-2 server deployment (protein embeddings)
- [ ] BrainLM server deployment (imaging embeddings)
- [ ] Integration tests for FM client

### 9.2 Data Modalities Not Yet Integrated

- [ ] Genomics: GWAS, PRS, SNP data
- [ ] Proteomics: Olink, SomaScan
- [ ] Metabolomics: NMR
- [ ] Imaging: MRI, retinal, DXA
- [ ] Accelerometer: Physical activity
- [ ] ECG: Cardiac electrophysiology
- [ ] Clinical text: Notes NLP

### 9.3 Advanced Analysis Features

- [ ] Multi-task learning across diseases
- [ ] Graph neural networks for comorbidity
- [ ] Time-series modeling (longitudinal)
- [ ] Causal inference (instrumental variables)
- [ ] Sensitivity analyses for unmeasured confounding

---

## 10. CONFIGURATION & SETUP

**File:** `.env`

```bash
# LLM Configuration
LLM_BASE_URL=https://api.openai.com  # Or local relay
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Multi-Model Routing (Optional)
MULTI_MODEL_ENABLED=true
AUTO_DISCOVER_MODELS=true
PREFERRED_MULTI_MODELS=gpt-5.4,gemini-3.1-pro-preview

# Data Paths
UKB_PARQUET_DIR=/path/to/parquet    # Fast path (priority)
UKB_RAW_DIR=/path/to/csv             # Fallback

# Biobank Identity (Configurable)
BIOBANK_NAME=UK Biobank
BIOBANK_ABBREVIATION=UKB
SUBJECT_ID_COL=eid
DIAGNOSES_CODE_COL=diag_icd10
DEATHS_CODE_COL=cause_icd10

# Optional Features
CUSTOM_SKILLS_DIR=./custom_skills
SEARCH_PROVIDER=duckduckgo
```

---

## 11. INTEGRATION POINTS FOR NOVEL APPROACHES

### 11.1 Where to Add Foundation Models

**Option A: Deploy FM Servers**
```python
# Implement RemoteFMClient in fm_embedding.py
class MyEvo2Client(FMEmbeddingInterface):
    def encode(self, data, modality, model_name=None):
        # Call Evo2 server: POST /v1/embeddings
        # modality='genomic' → send VCF/FASTA
        # Return (n_samples, 1024) numpy array
```

**Option B: Create Foundation Model Skills**
```python
@skill(
    name="embed_genomic",
    description="Extract genomic embeddings using Evo2 FM"
)
def embed_genomic(snp_list: str, *, ctx=None) -> dict:
    fm_client = RemoteFMClient(ctx.settings.fm_server_url)
    X_genomic = fm_client.encode(snp_data, modality='genomic')
    return {"embeddings": X_genomic.shape, "dims": 1024}
```

### 11.2 Where to Add Multi-Modal Analysis

**Extend `discover` pipeline:**
```python
# Step: Fuse multiple modalities
modalities = {
    'tabular': X_biomarkers,        # Blood tests
    'genomic': X_prs,               # PRS embeddings
    'imaging': X_brain_mri,         # BrainLM embeddings
    'protein': X_olink,             # ESM-2 embeddings
}
X_fused = fuse(modalities, method='attention')
model.fit(X_fused, y)
```

### 11.3 Where to Add Domain-Specific Skills

**Custom skill template:**
```bash
# Create new file
cat > custom_skills/omics_integration.py << 'SKILL'
from biobank_agent.registry import skill

@skill(
    name="integrate_omics",
    description="Integrate genomics + proteomics + metabolomics"
)
def integrate_omics(diseases: str, *, ctx=None) -> dict:
    # Your novel approach here
    pass
SKILL

# Automatically discovered at startup!
```

---

## 12. TESTING & VALIDATION

**File:** `tests/`

- 265 passing tests
- Marked with `@pytest.mark.integration` for API-dependent tests
- Run: `pytest tests/ -v -m "not integration"`

---

## 13. USAGE EXAMPLES

```bash
# Start agent
biobank

# Example queries
biobank> Discover disease-specific biomarkers for Type 2 Diabetes (E11)
biobank> Train an XGBoost model to predict E11 and show feature importance
biobank> Show Kaplan-Meier survival curves for acute MI (I21)
biobank> Run PheWAS for glucose (field 30740)
biobank> /plan Comprehensive cardiovascular risk analysis
biobank> Generate a Nature-quality report for all analyses
```

---

## SUMMARY TABLE: Current vs. Planned Capabilities

| Category | Current | Planned |
|----------|---------|---------|
| **Biomarkers** | Blood tests, anthropometrics, vitals | Genomics, proteomics, metabolomics, imaging |
| **ML Models** | XGBoost, LightGBM, CatBoost | Neural networks, foundation models |
| **Embeddings** | t-SNE, UMAP (scikit-learn) | Evo2, ESM-2, BrainLM (when deployed) |
| **Analysis Types** | Association, prediction, survival | Causal inference, multi-task learning |
| **Modalities** | Tabular only | Multimodal fusion with attention |
| **LLM** | OpenAI-compatible | Multi-model orchestration (debate mode) |
| **Figures** | Nature/ICML/NEJM/Lancet styles | Enhanced 3D visualization |

---

## References

- **Repository:** github.com/cpa2001/BioBank-Agent
- **Data:** UK Biobank Application 672073 (502K participants)
- **Technologies:** DuckDB, XGBoost, scikit-learn, lifelines, OpenAI API
- **License:** MIT

