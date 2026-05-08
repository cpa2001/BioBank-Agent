# UK Biobank Data Reference

Unified reference for all data modalities available in the UK Biobank dataset as accessed by this agent.

---

## Coverage Matrix

| Data Type | Participants | Coverage | Status | Access Method |
|-----------|--------------|----------|--------|---------------|
| Demographics | 502K | 100% | Ready | Parquet columns |
| Blood biomarkers | 480K | 95% | Ready | Biological_Samples.parquet |
| Metabolomics (NMR) | 487K | 97% | Ready | Biological_Samples.parquet |
| Exome sequencing | 469K | 93% | Ready | Genomics.parquet (metadata) |
| Brain MRI | 82K | 16% | Ready | Imaging derivatives |
| Cardiac MRI + ECG | 80–95K | 16–19% | Ready | Imaging derivatives |
| DXA (bone density) | 66K | 13% | Ready | Body composition fields |
| Retinal imaging | 44K | 9% | Ready | Fundus derivatives |
| Neurobiomarkers | 1.3K | 0.3% | Ready | COVID re-imaging subset |
| Proteomics (Olink) | ~53K | ~11% | Pending | Olink Explore 1536 |
| WGS | 150–300K | 30–60% | Varies | Per release |
| Accelerometer | 450K+ | 85%+ | Partial | Time-series files |

---

## Data Modalities Detail

### 1. Metabolomics (NMR)
- **Platform**: Nightingale Health 1H NMR
- **Location**: `Biological_Samples.parquet` (207 MB, 502K × 806 cols)
- **Metabolites**: ~509 unique measurements
- **Field IDs**: 20280–23948 (1,712 catalog entries)
- **Sample fields**: Field 20280 (Glucose-lactate, 487,279 participants), Field 23400 (Total Cholesterol, 488,513)
- **Subcategories**:
  - Glycolysis-related: glucose, lactate, pyruvate, citrate
  - Lipoproteins: LDL, HDL, VLDL subclasses and particle sizes
  - Lipids: phospholipids, cholesterol esters, triglycerides
  - Amino acids: alanine, glycine, histidine, isoleucine, leucine, valine, phenylalanine, tyrosine
  - Ketones: acetoacetate, beta-hydroxybutyrate
  - Fatty acids: monounsaturated, polyunsaturated, omega-3/6
  - Inflammatory: GlycA, glycation gap
- **Field categories**: Category 220 (249 fields — main measurements), Category 221 (249 — QC flags), Category 222 (10 — shipment/processing metadata)
- **Instances**: 4 per field (baseline + 3 follow-ups)

### 2. Blood Biomarkers
- **Location**: `Biological_Samples.parquet`
- **Tests**: 50+ routine clinical tests (blood counts, biochemistry, lipids, vitamins)
- **Coverage**: 85–95% depending on specific test
- **Subtypes**:
  - Routine blood counts (Category 100081): WBC, RBC, platelets, differentials (476K)
  - Biochemistry (Category 100080): lipids, liver, kidney, metabolic, hormones (420K–490K)
  - Saliva assays (Category 100082): cortisol and related markers
  - Urine assays (Category 100083): creatinine, electrolytes, proteins
  - Vitamins: Vitamin D, B12, folate (200K–300K)

### 3. Genomics
- **Location**: `Genomics.parquet` (371 MB, 502K × 194 cols)
- **Field IDs**: 23141–23172 (28 fields)
- **Exome**: Final OQFE release — 469,391 participants
- **WGS**: Variable by release (150K–300K)
- **Format**: Metadata in parquet; actual genotype files (VCF/PLINK/BGEN) in raw data

### 4. Brain Imaging
- **Coverage**: 16–17% (intentional subsample, ~82K)
- **Modalities**: T1 structural (20 fields), T2/FLAIR (5), dMRI (691), resting fMRI (45), task fMRI (26), SWI (36), brain atlases (1,000+)

### 5. Cardiac Imaging
- **Coverage**: 16–20% (79K–95K)
- **Types**: Heart MRI (10 fields), cardiac function derivatives (155), ECG (18), pulse wave analysis (23)

### 6. Bone Density (DXA)
- **Coverage**: 13% (66K)
- **Fields**: 168 total (body composition + bone density at hip, spine, forearm, whole body)

### 7. Abdominal Imaging
- **Coverage**: 13–40% (varies by organ)
- **Types**: Liver MRI, pancreas, kidney, organ composition
- **Pipelines**: AMRA, Calico, Uppsala

### 8. Retinal Imaging
- **Coverage**: 9% (44K)
- **Types**: Fundus images, optic disc, macular thickness (80+ derived fields)

### 9. Accelerometer / Physical Activity
- **Self-reported**: 500K+ (Fields 884, 894, 904, 914)
- **Wrist accelerometer**: Time-series activity counts (Field IDs ~90002–90100)

### 10. Neurobiomarkers
- **Coverage**: 0.25% (1.3K — COVID re-imaging subsample)
- **Biomarkers**: Amyloid-β40, Amyloid-β42, GFAP, NfL, pTau-181
- **Platform**: Quanterix Simoa HD-X
- **Field IDs**: 31040–31049

### 11. Proteomics
- **Platform**: Olink Explore 1536 (proximity extension assay, PEA)
- **Expected**: 1,400+ proteins
- **Expected coverage**: ~53,000 participants (with expanded ~200K via Data Portal)
- **Field IDs**: 30860–30903 (metadata); individual proteins via Olink NPX values
- **Panels**: Cardiovascular, Inflammation, Neurology, Oncology, Immunology
- **Format**: Normalized Protein Expression (NPX) values
- **Status**: Field definitions exist; full protein abundance data loading pending
- **Key metadata fields**: 30900 (proteins measured, 53,039), 30901 (plate), 30902 (well), 30903 (UKB-PPP selected, 6,230)

> **Note**: Fields 30860–30903 currently contain basic biochemistry in parquet, not the full Olink panel. Actual protein abundance data needs to be located in raw data directory.

---

## Field ID Quick Lookup

| Data Type | Field ID Range | Count |
|-----------|----------------|-------|
| Metabolites | 20280–23948 | 1,712 |
| Exome refs | 23141–23172 | 28 |
| Proteins (routine) | 30860–30903 | 33 |
| Brain MRI | 20100+ | 215+ |
| Blood counts | Various | 100+ |
| Biochemistry | 20000–30000 | 500+ |
| Neurobiomarkers | 31040–31049 | 10 |

---

## File Layout

```
$DATA_DIR/                          # Set via DATA_DIR in .env
├── ukb.parquet/                    # Main participant table (4 parts, 363 MB)
├── hesin_diag.parquet/            # Hospital episode diagnoses (6.9M records)
├── hesin_oper.parquet/            # Hospital episode operations
├── death_cause.parquet/           # Death records (112,917)
├── gp_clinical.parquet/           # General practitioner clinical records
├── gp_scripts.parquet/            # GP prescriptions
├── olink.parquet/                 # Proteomics (structure only — data pending)
├── categories/
│   ├── Biological_Samples.parquet  # Metabolomics + blood (806 cols, 207 MB)
│   ├── Genomics.parquet           # Exome + WGS metadata (194 cols, 371 MB)
│   ├── Health_Related_Outcomes.parquet  # Diagnoses (3,991 cols, 153 MB)
│   ├── Online_Follow_up.parquet   # Longitudinal (5,452 cols, 609 MB)
│   ├── Additional_Exposures.parquet  # Lifestyle (274 cols, 127 MB)
│   └── Population_Characteristics.parquet  # Demographics (30 cols, 10 MB)
├── sample_lists/                   # Participant sample definitions
├── field.txt                       # Field catalog (11,821 entries)
├── category.txt                   # Category definitions (410 categories)
├── esimpint.txt                   # Simple integer encoding lookup
└── ehierint.txt                   # Hierarchical encoding lookup
```

---

## Data Architecture

DuckDB serves as the unified query layer, reading from two tiers:
- **Tier 1 (fast):** Pre-built Parquet files — zero-copy column scans
- **Tier 2 (fallback):** Raw CSV files — lazy-loaded when a field isn't in Parquet

### Field Routing

When the agent queries a field, `DataManager.field_source()` routes to the fastest source:
1. **`biomarkers` view** — core parquet (484 fields, fastest)
2. **`cat_*` views** — category parquets (4,487 fields, fast)
3. **`csv_*` views** — raw CSV fallback (lazy-loaded, slow but comprehensive)

---

## Dataset Statistics

### Core Biomarkers (`ukb.parquet`)

| Property | Value |
|----------|-------|
| Rows | 502,370 |
| Columns | 2,031 (484 unique field IDs) |
| Size | 363 MB (4 SNAPPY-compressed partitions) |
| Format | Row-partitioned, `{field_id}-{instance}.{array}` column names |

### Clinical Records

| Dataset | Rows | Columns | Size |
|---------|------|---------|------|
| `hesin_diag.parquet` (ICD10 diagnoses) | 6,946,795 | 4 | 37 MB |
| `death_cause.parquet` (cause of death) | 112,917 | 5 | 576 KB |

### Category Parquets (`categories/`)

Built from raw CSVs via `biobank rebuild-parquet`. Each category is a separate file.

| Category | Fields | Columns | Size |
|----------|--------|---------|------|
| Population_Characteristics | 29 | 30 | 10 MB |
| Biological_Samples | 437 | 806 | 207 MB |
| Additional_Exposures | 240 | 274 | 127 MB |
| Genomics | 131 | 194 | 371 MB |
| Health_Related_Outcomes | 2,550 | 3,991 | 153 MB |
| Online_Follow_up | 1,100 | 5,452 | 609 MB |
| **Total** | **4,487** | **10,747** | **1.55 GB** |

**Grand total: 4,971 unique field IDs** (484 core + 4,487 category).

### Raw CSV Sources

All under `$RAW_DIR/UKB_info/`:

| File | Columns | Size |
|------|---------|------|
| `ukb672073_Population_Characteristics.csv` | 34 | 78 MB |
| `ukb672073_Biological_Samples.csv` | 1,775 | 3.5 GB |
| `ukb672073_Additional_Exposures.csv` | 279 | 679 MB |
| `ukb672073_Genomics.csv` | 194 | 1.0 GB |
| `ukb672073_Health_Related_Outcomes.csv` | 4,896 | 7.3 GB |
| `ukb672073_Online_Follow_up.csv` | 5,467 | 8.3 GB |

Main dataset (`$RAW_DIR/UKB/ukb672073.csv`): 30,799 columns, 54 GB.

---

## Key Biomarker Groups

### Blood Biochemistry (30 fields: 30600–30900)

Albumin, ALT, AST, ApoA, ApoB, Bilirubin, Calcium, Cholesterol, Creatinine, CRP, Cystatin C, GGT, Glucose, HbA1c, HDL, IGF-1, LDL, Lipoprotein A, Phosphate, SHBG, Testosterone, Total protein, Triglycerides, Urate, Urea, Vitamin D

### Blood Count (31 fields: 30000–30300)

WBC, RBC, Haemoglobin, Haematocrit, MCV, MCH, MCHC, RDW, Platelets, Lymphocytes, Monocytes, Neutrophils, Eosinophils, Basophils, Reticulocytes

### Demographics (6 fields)

Sex (31), Year of birth (34), Month of birth (52), Ethnic background (21000), Age at recruitment (21003/21022)

### Anthropometric (5 fields)

Waist circumference (48), Hip circumference (49), Standing height (50), BMI (21001/23104)

---

## Top 10 Diagnoses (by Patient Count)

| Rank | Code | Disease | N | Prevalence |
|------|------|---------|---|------------|
| 1 | I10 | Essential hypertension | 159,860 | 31.8% |
| 2 | Z86 | Personal history of conditions | 128,975 | 25.7% |
| 3 | Z92 | Personal history of treatment | 90,965 | 18.1% |
| 4 | E78 | Lipoprotein metabolism disorders | 82,940 | 16.5% |
| 5 | Z87 | Family history of diseases | 69,770 | 13.9% |
| 6 | M54 | Dorsalgia | 62,741 | 12.5% |
| 7 | K57 | Diverticular disease | 61,424 | 12.2% |
| 8 | J06 | Upper respiratory infections | 50,424 | 10.0% |
| 9 | I25 | Chronic ischaemic heart disease | 47,089 | 9.4% |
| 10 | E11 | Type 2 diabetes | 43,769 | 8.7% |

## Death Records

- 44,198 deceased participants (8.8% mortality rate)
- 112,917 total cause-of-death records
- Top causes: I259 (chronic ischaemic heart disease), C349 (lung cancer), J189 (pneumonia)

---

## Metadata Files

| File | Records | Size | Description |
|------|---------|------|-------------|
| `field.txt` | 11,821 | 4 MB | Field ID → name, category, value type |
| `category.txt` | 410 | 172 KB | Category ID → name |
| `esimpint.txt` | 18,079 | 649 KB | Simple integer encoding lookup |
| `ehierint.txt` | 28,901 | 1.4 MB | Hierarchical encoding lookup |

---

## Best Use Cases

### High-Power Studies (N > 400K)
- Metabolomics biomarker discovery (487K × 509 metabolites)
- Blood biomarker + disease outcome associations
- Multi-omics integration (metabolomics + genomics overlap ~400K)

### Imaging + Biomarker Studies (N ~ 60–82K)
- Brain MRI + metabolomics correlations
- Cardiac MRI + blood biomarker risk scores
- DXA + blood biochemistry for fracture risk

### Genetic Studies
- Exome-phenotype associations (469K)
- PRS validation (485K)
- Multi-biomarker GWAS

---

## Imaging Detail

### Brain MRI

| Modality | Category | Fields | Participants | Details |
|----------|----------|--------|--------------|---------|
| T1 structural | 110 | 20 | 82,234 | High-resolution anatomy, volumetrics |
| T2/FLAIR | 112 | 5 | 80,371 | Pathology detection, white matter lesions |
| Diffusion (dMRI) | 107, 134, 135 | 691 | 79,381 | Tract-tracing, FA, MD, AD, RD |
| Susceptibility (SWI) | 109 | 36 | 74,141 | Iron, microbleeds, vasculature |
| Arterial Spin Labeling | 119 | 51 | 8,865 | Cerebral blood flow |
| Resting fMRI | 111 | 45 | 82,105 | BOLD connectivity, intrinsic networks |
| Task fMRI | 106 | 26 | 67,499 | Hariri faces/shapes emotion task |
| Freesurfer A2009S atlases | 197 | 444 | ~83K | Largest single category |
| Surface-based fMRI | 198 | varies | ~83K | Resting & task surface measures |
| Native atlases | 200 | varies | ~40K | Native space parcellations |
| Diffusion tractography | 201 | varies | ~79K | White matter tracts |
| Structural/functional connectivity | 202 | varies | ~79K | Connectomes |
| Connectome networks | 204 | varies | ~79K | Structural/functional networks |

**Total brain imaging fields**: ~880+

### Cardiac Imaging

| Type | Category | Fields | Participants | Details |
|------|----------|--------|--------------|---------|
| Heart MRI | 102 | 10 | 79,940 | Cardiac anatomy and structure |
| Cardiac function (Petersen) | 157 | 83 | 79,320 | Aortic function methodology |
| Cardiac function (Biasiolli) | 162 | 72 | 10,908 | Alternative methodology |
| LV size and function | 133 | varies | ~79K | Left ventricular phenotypes |
| Cardiac classifications | 523 | varies | ~79K | Clinical image-derived phenotype |
| Pulse wave analysis | 128 | 23 | ~80K | Blood pressure + waveform (supine) |
| 12-lead ECG at rest | 104 | 18 | 95,000–101,000 | Fields 12323–20205 |

**Total cardiac fields**: ~150+

### Abdominal Imaging

| Organ | Category | Fields | Participants |
|-------|----------|--------|--------------|
| Liver MRI | 126 | 11 | 39K |
| Pancreas MRI | 131 | 4 | Not enumerated in the current local catalogue snapshot |
| Kidney MRI | 156 | 4 | Not enumerated in the current local catalogue snapshot |
| General abdominal | 105 | 5 | 79,524 |
| AMRA body composition | 149 | 32 | 21,975 |
| Calico pipeline | 158 | 19 | varies |
| Uppsala pipeline | 159 | 5 | kidney focused |

### Bone Density (DXA)

| Category | Fields | Participants | Details |
|----------|--------|--------------|---------|
| DXA assessment | 103 | 6 | 66,856 | Heel ultrasound (Field 19) |
| Body composition by DXA | 124 | 46 | 52,632 | Fat mass, lean mass, bone-free mass |
| Bone density/minerals | 125 | 94 | 57,045 | BMD T-scores (hip, spine, forearm) |
| DXA-derived skeletal proportions | 522 | varies | varies | Skeletal geometry |

**Total DXA fields**: ~180+

### Retinal Imaging

| Category | Fields | Participants | Details |
|----------|--------|--------------|---------|
| Retinal grading derived OCT | 1080 | 38 | 17,935 | Manually graded qualitative features |
| Extended derived OCT | 1081 | 8 | 33,482 | Quantitative OCT (Han et al.) |
| Macular thickness | 100079 | 80 | 61,883 | Layer measurements |
| Retinal OCT scans | 725 | varies | varies | 3D OCT scan data |
| Colour Fundus-derived | 521 | varies | varies | Quantitative retinal features |
| 3D Retinal OCT | 100016 | varies | varies | Full OCT volumes |

**Total retinal fields**: ~100+

---

## Genomics Detail

### Exome Sequencing (Category 170–171)
- **Field IDs**: 23141–23172 (28 fields)
- **Final release (OQFE)**: 469,391 participants
- **Interim 450K**: 454,372
- **Interim 300K**: 301,989
- **Interim 200K**: 200,414
- **Initial 50K**: 49,925
- **Format**: VCF, PLINK, BGEN
- **Protocol**: OQFE (Functionally Equivalent with original quality scores)

### Whole Genome Sequencing (Category 180–187)
- **Field IDs**: 23180+
- **Status**: Limited to specific cohorts (~150K–500K)
- **Formats**: CRAM, VCF, PLINK

### Special Genomic Data
- **Field 23165**: Blood-type haplotypes (487K participants)
- **Population-level PRS**: Categories 301–302 (~485K participants)

> **Note**: Field IDs in Genomics.parquet reference file locations (VCF/CRAM/PLINK), not array data. Actual genotype matrices stored in separate file system.

---

## Activity & Accelerometer Detail

### Self-Reported Physical Activity
| Field | Description | Participants |
|-------|-------------|--------------|
| 884 | Days/week moderate activity | 501,099 |
| 894 | Duration moderate activity | 423,529 |
| 904 | Days/week vigorous activity | 501,099 |
| 914 | Duration vigorous activity | 312,663 |
| 6164 | Types of activity in last 4 weeks | 496,916 |

### Wrist-Worn Accelerometer
- **Expected field IDs**: 90002–90100 range
- **Data format**: Time-series activity counts (per-hour or per-minute)
- **Coverage**: Estimated 85%+
- **Status**: File location needs verification

---

## Data Quality Summary

### High Coverage (>90%)
- Metabolomics: 97% (487K/502K)
- Blood biomarkers: 85–95% (varies)
- Genetic data: 93% (469K/502K exome)
- Physical activity (questionnaire): 99.8% (501K/502K)
- Demographics: 99.9%

### Intentional Subsamples (10–20%)
- Brain MRI: 16–17% (82K)
- Heart MRI: 16% (79K)
- DXA: 13% (66K)
- ECG: 19% (95K)
- Retinal: 9% (44K)

### Pending / Low Coverage
- Neurobiomarkers: 0.25% (1.3K — COVID re-imaging substudy)
- Olink proteomics: Unknown (data not yet loaded)
- Accelerometer time-series: ~85%+ (not mapped in the current local parquet layout)

---

## Participant Demographics

| Property | Value |
|----------|-------|
| Total participants | 502,372 |
| Age at recruitment | 40–69 years (2006–2010) |
| Ethnicity | ~94% White British |
| Sex | ~55% female, ~45% male |
| Deceased | 44,198 (8.8% mortality rate) |

### Time Points (Instances)

Most fields have up to 4 instances:
- **Instance 0**: Baseline assessment (2006–2010)
- **Instance 1**: First follow-up (~2012–2013)
- **Instance 2**: Imaging-focused follow-up (~2014–2015)
- **Instance 3**: Online/latest follow-up (2020+)

Column naming: `{field_id}-{instance}.{array}` (e.g. `30000-0.0`, `30000-1.0`)

### Array Indices

Some fields have multiple measurements per instance:
- Ultrasound: up to 6 measurements (Field 84)
- Arterial stiffness: up to 34 measurements (Field 87)
- Brain imaging: up to 444 measurements (Category 197)

---

## Blood Biomarker Field IDs

### Blood Biochemistry (Category 100080, Fields 30600–30890)

| Field | Biomarker | Participants |
|-------|-----------|--------------|
| 30600 | Albumin | ~470K |
| 30620 | ALT (Alanine aminotransferase) | ~470K |
| 30650 | AST (Aspartate aminotransferase) | ~470K |
| 30660 | Direct bilirubin | ~470K |
| 30670 | Urea | ~470K |
| 30680 | Calcium | ~470K |
| 30690 | Cholesterol | ~470K |
| 30700 | Creatinine | ~470K |
| 30710 | C-reactive protein | ~470K |
| 30720 | Cystatin C | ~470K |
| 30730 | GGT (Gamma glutamyltransferase) | ~470K |
| 30740 | Glucose | ~470K |
| 30750 | HbA1c | ~470K |
| 30760 | HDL cholesterol | ~470K |
| 30770 | IGF-1 | ~470K |
| 30780 | LDL cholesterol (direct) | ~470K |
| 30790 | Lipoprotein A | ~350K |
| 30800 | Oestradiol | ~470K |
| 30810 | Phosphate | ~470K |
| 30830 | SHBG | ~470K |
| 30840 | Total bilirubin | ~470K |
| 30850 | Testosterone | ~470K |
| 30860 | Total protein | ~431K |
| 30870 | Triglycerides | ~470K |
| 30880 | Urate | ~470K |
| 30890 | Vitamin D | ~470K |

### Blood Count (Category 100081, Fields 30000–30300)

| Field | Marker | Participants |
|-------|--------|--------------|
| 30000 | WBC (White blood cell count) | ~476K |
| 30010 | RBC (Red blood cell count) | ~476K |
| 30020 | Haemoglobin | ~476K |
| 30030 | Haematocrit | ~476K |
| 30040 | MCV (Mean corpuscular volume) | ~476K |
| 30050 | MCH (Mean corpuscular haemoglobin) | ~476K |
| 30060 | MCHC | ~476K |
| 30070 | RDW (Red cell distribution width) | ~476K |
| 30080 | Platelet count | ~476K |
| 30120 | Lymphocyte count | ~476K |
| 30130 | Monocyte count | ~476K |
| 30140 | Neutrophil count | ~476K |
| 30150 | Eosinophil count | ~476K |
| 30160 | Basophil count | ~476K |
| 30240 | Reticulocyte count | ~476K |
| 30250 | Reticulocyte percentage | ~476K |
| 30280 | Immature reticulocyte fraction | ~476K |

### Other Biological Assays
- **Category 100082**: Saliva assays — Cortisol and related markers
- **Category 100083**: Urine assays — Creatinine, electrolytes, proteins

---

## Genetic Data Detail

### Whole Genome Genotyping (Category 263)

| Component | Fields | Details |
|-----------|--------|---------|
| SNP array calls | 22000–22009 | Affymetrix BiLEVE & Biobank arrays |
| QC metrics | 22010–22051 | Missingness, HWE, MAF, heterozygosity, relatedness |
| Imputed variants | 26200–26290 | 96M+ SNPs |
| Genetic PCs | Various | Ancestry components |
| HLA imputation | Various | Classical HLA types |

- **Array types**: UK BiLEVE Axiom (~50K), UK Biobank Axiom (~450K)
- **Genome build**: GRCh37
- **Coverage**: ~488,000 participants (97% of cohort)
- **Core SNP markers**: 805,426

### Whole Exome Sequencing (Categories 170, 172)

| Release | Participants | Date |
|---------|-------------|------|
| 50K pilot | 49,925 | March 2019 |
| 200K | 200,414 | 2020 |
| 300K | 301,989 | 2021 |
| 450K | 454,372 | 2022 |
| Final (OQFE) | 469,391 | Current |

- **Capture**: IDT xGen Exome Research Panel v1.0 (39 Mbp, 19,396 genes)
- **Platform**: Illumina NovaSeq 6000 (75×75 bp paired-end)
- **Coverage**: 95.2% of targeted bases > 20X
- **Processing**: DeepVariant variant calling, GLnexus joint-genotyping
- **Formats**: CRAM, gVCF, multi-sample VCF (pVCF), PLINK

---

## Additional Modalities

### Cardiovascular & Arterial Function

| Type | Category | Method | Key Fields |
|------|----------|--------|------------|
| Arterial stiffness | 100007 | PulseTrace PCA2 (finger) | Field 87 (up to 34 indices) |
| Pulse wave analysis | 128 | Heart MRI supine | Blood pressure + waveform |
| ECG at rest | 104 | 12-lead | Fields 12323–20205 |
| ECG during exercise | 100012 | Cycle ergometry (4-lead) | Pre-test, activity, recovery |

### Cognitive Assessment (Categories 501–505)

| Category | Test | Description |
|----------|------|-------------|
| 501 | Matrix pattern completion | Non-verbal reasoning |
| 502 | Symbol digit substitution | Processing speed |
| 503 | Tower rearranging | Executive function |
| 504 | Picture vocabulary | Verbal ability |
| 505 | Trail making | Attention/switching |

### Disease Outcomes (Categories 2401–2417)

- **Category 2401**: Algorithmically-defined health outcomes (346 fields)
  - Stroke (Cat 43), MI (Cat 44), Asthma (Cat 45), COPD (Cat 46)
  - Dementia (Cat 47), ESRD (Cat 48), and 30+ other outcomes
- **Categories 2404–2415**: ICD-10 diagnosis codes
- **Categories 2416–2417**: OPCS procedural coding
- **File**: Health_Related_Outcomes.parquet (153 MB, 3,991 columns)

### Longitudinal Follow-Up

- **File**: Online_Follow_up.parquet (609 MB, 5,452 columns)
- **Contents**: Hospital admissions (20077+), medical history updates (20078+), medication changes (20099+)
- **Coverage**: 60–80% of participants

### Lifestyle & Environmental Exposure

| Domain | Key Fields | Coverage |
|--------|------------|----------|
| Dietary patterns | 21100–21106 | 40–70% |
| Alcohol intake | 1558 | ~95% |
| Smoking status | 20116 | ~95% |
| Pack-years | 20161 | ~40% |
| Sleep duration | 1160 | ~95% |
| Air pollution | Category 114–115 | varies |
| Greenspace | Category 151 | varies |
| Occupational | 24003–24665 | varies |

### Mental Health (Categories 210, 517–519)

| Category | Assessment | Description |
|----------|-----------|-------------|
| 210 | Berlin Questionnaire | Sleep disturbances |
| 517 | Adult SWAN Rating Scale | ADHD |
| 519 | Affective Reactivity Index | Emotional dysregulation |

### Cardiorespiratory Fitness

- **Category 267**: VO2max during exercise (cycle ergometry)

### Accelerometer Data (Categories 1008–1009, 1020)

| Category | Type | Details |
|----------|------|---------|
| 1008 | Raw acceleration | Wrist-worn, June 2013 – Jan 2016, ~100K participants |
| 1009 | Acceleration averages | Aggregated metrics, ±8g truncation |
| 1020 | Derived phenotypes | Walmsley et al. BJSM 2022 — sedentary/light/moderate/vigorous |

- **Device**: Wrist-worn accelerometer
- **Seasonal repeats**: Quarterly follow-ups (2018+)

---

## Field ID Master Reference by Modality

| Modality | Field ID Range | Category | Count |
|----------|----------------|----------|-------|
| Procedural metrics | 3–6 | 152 | 4 |
| Blood pressure | 50–55, 93–94, 4079–4080 | 100011 | 7 |
| Anthropometry | 21, 48–50, 23104 | 100010 | 5 |
| DXA | 19, 77–96 | 103, 124–125 | 20+ |
| Blood counts | 30000–30300 | 100081 | 31 |
| Blood chemistry | 30600–30890 | 100080 | 30 |
| Cardiac MRI | Various | 102, 157, 162 | 150+ |
| Brain MRI (all) | Various | 100, 106–112, 119 | 880+ |
| Retinal OCT | Various | 100016, 521, 725, 1080–1081 | 100+ |
| Genetics | 21007–26290 | 263, 170, 172 | 193+ |
| Disease outcomes | 27980–28038 | 2401–2417 | 319+ |
| Follow-up | 20077–20999 | 135, 220–221 | 243+ |
| Exposures | 21100–24665 | 151, 220 | 363 |
| Accelerometry | Various | 1008–1009, 1020 | 40+ |
| Proteomics (NPX) | 30900+ | 1838–1839 | 5,000+ |

---

## Not Present in This Dataset

- Raw imaging files (DICOM, NIfTI) — only derived phenotypes
- Free text medical records — only coded/structured data
- Audio/video recordings
- Raw wearable sensor data (beyond accelerometer)
- Genomic sequence files (FASTQ/BAM) — field references only
- Microbiome data

---

## Key Statistics Summary

| Metric | Value |
|--------|-------|
| Total participants | 502,372 |
| Total UKB field codes | ~18,500 |
| Unique fields indexed | 4,971 (484 core + 4,487 category) |
| Total data points | >5.3 billion (502K × 18.5K) |
| Raw data size | 149 GB |
| Indexed Parquet size | 1.55 GB |
| Categories | 410 distinct measurement categories |
| Imaging-derived fields | ~1,560 |
| NMR metabolites | 509 |
| Hospital diagnoses | 6,946,795 records |
| Death records | 112,917 |
| Assessment centres | 22 |

### Missing Data Characteristics
- Varies by modality: 1–99% depending on field
- Demographics: 95–99% complete
- Lifestyle: 40–70% complete
- Imaging: intentional 10–20% subsamples
- Encoded as empty strings in CSV files

---

## Codebase Integration

### Data Features Mapped (`biobank_agent/data/features.py`)
- Blood biochemistry (30 fields)
- Blood count (31 fields)
- Anthropometric (5 fields)
- Blood pressure (5 fields)
- Demographics (6 fields)
- Lifestyle (4 fields)

### Modality-Specific Skills
| Skill | File | Modalities Used |
|-------|------|-----------------|
| `biomarker_dist` | skills/biomarker_dist.py | Blood biochemistry, NMR |
| `correlation` | skills/correlation.py | Cross-modality |
| `discovery` | skills/discovery.py | All modalities |
| `embedding` | skills/embedding.py | Tabular + imaging |
| `survival` | skills/survival.py | Outcomes + exposures |
| `gwas_proxy` | skills/gwas_proxy.py | Genomics |
| `phewas` | skills/phewas.py | All phenotypes |
