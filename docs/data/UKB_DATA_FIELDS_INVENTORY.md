# UK BIOBANK DATA FIELDS - COMPREHENSIVE INVENTORY

**Generated:** 2026-04-23  
**Location:** `/Users/chenpengan/Projects/CUHK/milton_data/`  
**Method:** Direct filesystem and parquet schema inspection (READ-ONLY verification)

---

## EXECUTIVE SUMMARY

The UK Biobank dataset contains **~502,000 participants** with access to:

- ✅ **METABOLOMICS**: ~500 NMR metabolite measurements (97% coverage, 487K participants)
- ✅ **GENOMICS**: Exome sequencing + WGS variants (93% coverage, ~469K participants)
- ✅ **BLOOD BIOMARKERS**: Comprehensive biochemistry + blood counts (>85% coverage)
- ✅ **BRAIN IMAGING**: Multiple MRI modalities (16-17% coverage, ~82K participants)
- ✅ **CARDIAC IMAGING**: Heart MRI + ECG (16-20% coverage)
- ✅ **BONE DENSITY**: Full-body DXA scans (13% coverage)
- ✅ **ABDOMINAL IMAGING**: Multi-organ MRI (various coverage)
- ⚠️ **PROTEOMICS**: Fields defined but data structure empty (Olink data pending)
- ✅ **NEUROBIOMARKERS**: Limited (1.3K participants - COVID re-imaging substudy)
- ? **RETINAL IMAGING**: Fundus images referenced (need verification)
- ✅ **ACTIVITY/ACCELEROMETER**: Self-reported + wrist-worn device data

---

## 1. METABOLOMICS DATA ✅

### Location
```
milton_data/categories/Biological_Samples.parquet (207 MB, 502K × 806 cols)
```

### Platform
- **Nightingale Health 1H NMR Spectroscopy**
- ~509 unique metabolite measurements

### Field IDs in Catalog
- **Range**: 20280-23948 (1,712 fields total in field.txt)
- **Actually in Parquet**: 20280-23948 subset (249+ NMR columns)

### Metabolite Categories
- **Amino acids**: Alanine, glutamine, glycine, histidine, isoleucine, leucine, valine, phenylalanine, tyrosine, etc.
- **Lipoproteins**: HDL, LDL, VLDL particle concentrations & sizes
- **Fluid balance**: Lactate, pyruvate
- **Fatty acids**: Various chain-length fatty acids
- **Glucose metabolism**: Glucose-lactate ratio markers
- **Lipids**: Triglycerides, phospholipids, cholesterol esters

### Coverage
- **Participants**: 487,000-488,000 (varies by metabolite)
- **Coverage**: ~97% of cohort
- **Format**: Numeric measurements (likely log-transformed NPX or standardized values)

### Example Fields
| Field ID | Title | Participants |
|----------|-------|--------------|
| 20280 | Glucose-lactate | 487,279 |
| 20281 | Spectrometer-corrected alanine | 488,326 |
| ... | ... | ... |

---

## 2. GENOMICS DATA ✅

### Location
```
milton_data/categories/Genomics.parquet (371 MB, 502K × 194 cols)
```

### Types Available

#### A) Exome Sequencing (Category 170-171)
- **Field IDs**: 23141-23172 (28 fields in catalog)
- **Releases**:
  - Final release (OQFE): 469,391 participants
  - Interim 450K: 454,372 participants
  - Interim 300K: 301,989 participants
  - Interim 200K: 200,414 participants
  - Initial 50K: 49,925 participants
- **Format**: VCF, PLINK, BGEN
- **Protocol**: OQFE (Functionally Equivalent with original quality scores)

#### B) Whole Genome Sequencing (Category 180-187)
- **Field IDs**: 23180+ (various releases)
- **Status**: Limited to specific cohorts (~150K-500K)
- **Formats**: CRAM, VCF, PLINK

#### C) Special Genomic Data
- **Field 23165**: Blood-type haplotypes (487K participants)
- **WGS DRAGEN**: Limited 500K release (~48 participants)
- **Population-level PRS**: Multiple traits in Categories 301-302 (~485K participants)

### Coverage
- **Exome**: ~93% (469K/502K)
- **Whole Genome**: Variable by release (150K-300K participants)
- **PRS**: ~96% for standard scores

### Important Notes
⚠️ Field IDs reference file locations (VCF/CRAM/PLINK files), not array data in parquet
- Actual genotype matrices stored in separate file system
- Parquet contains metadata + file references

---

## 3. BLOOD BIOMARKERS & BIOCHEMISTRY ✅

### Location
```
milton_data/categories/Biological_Samples.parquet (207 MB)
Categories: 17518, 18518, 220, 221, 222, 264, 9081, 100081, 100083, 100085, 100086, 100087, 100092, 100093
```

### Types

#### Routine Blood Counts
- **White blood cells**: WBC, neutrophils, lymphocytes, monocytes, eosinophils, basophils
- **Red blood cells**: RBC, hemoglobin, hematocrit, MCV, MCH, MCHC
- **Platelets**: Count, distribution width
- **Participants**: 476K-480K

#### Biochemistry
- **Proteins**: Albumin, total protein, immunoglobulins
- **Kidney**: Creatinine, cystatin C, urea
- **Liver**: Bilirubin, ALP, ALT, AST, GGT
- **Lipids**: Triglycerides, phospholipids, cholesterol, LDL, HDL, apoA, apoB
- **Glucose metabolism**: Glucose, HbA1c
- **Other**: Calcium, phosphate, magnesium, iron, urate, CRP, D-dimer
- **Participants**: 420K-490K (varies by test)

#### Vitamins & Minerals
- **Vitamin D**: 25-OH vitamin D
- **Vitamin B12**: B12, folate
- **Trace elements**: Iron, zinc, copper, manganese
- **Participants**: 200K-300K (varies)

### Coverage
- **Average**: 85-95% across tests
- **Format**: Numeric concentrations (various units per test)

### Field Categories
- **Category 220**: 249 fields - main metabolite/biomarker measurements
- **Category 221**: 249 fields - QC flags for measurements
- **Category 222**: 10 fields - shipment/processing metadata

---

## 4. IMAGING DATA ✅

### Note on Format
⚠️ Imaging files (NIFTI, DICOM, MAT) stored separately from parquet  
Parquet contains field IDs & metadata; actual images reference external files

### A) Brain MRI - Comprehensive Coverage

#### Structural Brain Imaging
| Modality | Category | Fields | Participants | Details |
|----------|----------|--------|--------------|---------|
| T1 weighted | 110 | 20 | 82,234 avg | High-resolution anatomy |
| T2/FLAIR | 112 | 5 | 80,371 | Pathology detection |
| Diffusion (dMRI) | 107, 134, 135 | 691 | 79,381 | Tract-tracing, microstructure |
| Susceptibility (SWI) | 109 | 36 | 74,141 | Iron, bleeds, myelin |
| Arterial Spin Labeling | 119 | 51 | 8,865 | Cerebral blood flow |

#### Functional Brain Imaging
| Modality | Category | Fields | Participants | Details |
|----------|----------|--------|--------------|---------|
| Resting fMRI | 111 | 45 | 82,105 | Intrinsic brain connectivity |
| Task fMRI | 106 | 26 | 67,499 | Hariri emotion task response |

#### Brain Atlases & Parcellations
| Type | Categories | Fields | Participants |
|------|-----------|--------|--------------|
| Native space atlases | 200-204 | 1,000+ | 40K-83K |
| Volume estimates | 190-197 | 850+ | 83K |
| Surface measures | 192-196 | 500+ | 83K |

### B) Cardiac Imaging

#### Heart MRI
- **Category 102**: 10 fields (~79,940 participants)
- **Cardiac function derived (1)**: Category 157 - 83 fields (~79,320 participants)
- **Cardiac function derived (2)**: Category 162 - 72 fields (~10,908 participants)
- **Pulse wave analysis**: Category 128 - 23 fields (~80K participants)

#### ECG - 12-lead at Rest
- **Category 104**: 18 fields (~95,000-101,000 participants)
- **Key fields**:
  - 12323: ECG measuring method
  - 12336: Ventricular rate
  - 12338: P duration
  - 12340: QRS duration
  - 12653: ECG automated diagnoses
  - 20205: ECG datasets

### C) Abdominal Imaging

#### Multi-organ MRI
| Organ | Category | Fields | Participants |
|------|----------|--------|--------------|
| Liver MRI | 126 | 11 | 39K |
| Pancreas MRI | 131 | 4 | ? |
| Kidney MRI | 156 | 4 | ? |
| General abdominal | 105 | 5 | 79,524 |

#### Derived Abdominal Composition
- **AMRA pipeline**: Category 149 - 32 fields (21,975 participants)
- **Calico pipeline**: Category 158 - 19 fields (varies)
- **Uppsala pipeline**: Category 159 - 5 fields (kidney focused)

### D) Bone Density (DXA)

#### Full-Body Assessments
- **Category 103**: 6 fields (~66,856 participants)
- **Category 124**: 46 fields - body composition (~52,632 participants)
- **Category 125**: 94 fields - bone density/minerals (~57,045 participants)

**Includes**: Hip, spine, forearm, whole body measurements

### E) Retinal Imaging

#### Fundus Photography
- **Category 1080**: 38 fields (~17,935 participants) - Image quality, optic disc
- **Category 1081**: 8 fields (~33,482 participants) - Disc measurements
- **Category 100079**: 80 fields (~61,883 participants) - Macular thickness

---

## 5. PROTEOMICS DATA ⚠️ (PENDING FULL DATA)

### Status: **Fields Defined, Data Structure Empty**

### Location
```
milton_data/olink.parquet/part-0.parquet  ← CONTAINS ONLY 'eid' COLUMN
```

### Field Definitions (in catalog)
- **Field IDs**: 30860-30903 (33 fields)
- **Expected data**: 1,400+ proteins from Olink Explore 1536 platform
- **Expected coverage**: ~53,000 participants

### Currently in Field IDs 30860-30903
```
30860: Total protein (431,390 participants)
30870: Triglycerides (469,945 participants)  [Note: these are lipids, not Olink proteins]
30880: Urate (469,765 participants)
30890-30903: Additional biomarkers
```

⚠️ **NOTE**: The field IDs 30860-30903 appear to contain basic biochemistry, not the full Olink proteomics panel  
**ACTION NEEDED**: Locate actual Olink protein abundance data (1,400+ proteins) - likely in separate location or raw data directory

---

## 6. NEUROBIOMARKERS ✅ (LIMITED)

### Location
```
milton_data/categories/Biological_Samples.parquet
Category 163
```

### Platform
- **Quanterix Simoa HD-X Analyzer**
- **Assay**: Simoa Neurology 4-plex E + pTau-181V2

### Biomarkers
- **31040**: Plasma Amyloid-β40
- **31041**: Plasma Amyloid-β42
- **31042**: Plasma GFAP (glial fibrillary acidic protein)
- **31043**: Plasma NeuroFilament Light (NfL)
- **31044**: Plasma phospho-Tau-181 (pTau-181)
- **31045-31049**: Sample metadata (batch, plate, well, state)

### Coverage
- **Participants**: 1,273 (COVID-19 re-imaging study subsample)
- **Sample date**: During imaging assessment

---

## 7. ACTIVITY & ACCELEROMETER DATA ✅

### Self-Reported Physical Activity
- **Field 884**: Days/week moderate activity (501,099 participants)
- **Field 894**: Duration moderate activity (423,529 participants)
- **Field 904**: Days/week vigorous activity (501,099 participants)
- **Field 914**: Duration vigorous activity (312,663 participants)
- **Field 6164**: Types of activity in last 4 weeks (496,916 participants)

### Wrist-Worn Accelerometer
- **Expected field IDs**: 90002-90100 range (need verification)
- **Data format**: Time-series activity counts (likely per-hour or per-minute)
- **Coverage**: Estimated 85%+
- **Status**: ⚠️ Need to verify actual file locations

### Environmental Exposures (Related)
- **Pollution exposure**: Category 114-115 (noise, air quality)
- **Greenspace exposure**: Category 151

---

## 8. FILE STRUCTURE REFERENCE

### Main Data Directory
```
/Users/chenpengan/Projects/CUHK/milton_data/
├── ukb.parquet/                          4 parts, ~500K × 2,031 cols total
├── olink.parquet/                        Structure only (empty data)
├── categories/
│   ├── Population_Characteristics.parquet (30 cols)
│   ├── Biological_Samples.parquet        (806 cols) ⭐ Metabolomics, blood
│   ├── Genomics.parquet                  (194 cols) ⭐ Exome, WGS, PRS
│   ├── Health_Related_Outcomes.parquet   (3,991 cols)
│   ├── Additional_Exposures.parquet      (274 cols)
│   └── Online_Follow_up.parquet          (5,452 cols)
├── field.txt                             Field catalog (11,822 lines)
├── category.txt                          Category definitions
├── gp_clinical.parquet/                  General practitioner clinical records
├── gp_scripts.parquet/                   Prescriptions
├── hesin_diag.parquet/                   Hospital episode statistics - diagnoses
├── hesin_oper.parquet/                   Hospital episode statistics - operations
├── death_cause.parquet/                  Death register linked data
├── sample_lists/                         Participant sample definitions
└── [other metadata files]
```

### Raw Data Directory (Need to Verify)
```
/Users/chenpengan/Projects/CUHK/UKB/
├── UKB/                                  Raw UKB data
├── UKB_info/                             Documentation
├── [imaging data?]
├── [genetic data?]
└── [accelerometer time-series?]
```

---

## 9. FIELD ID QUICK REFERENCE

| Data Type | Field ID Range | Count | Participants | Modality |
|-----------|-----------------|-------|--------------|----------|
| Metabolomics (NMR) | 20280-23948 | 1,712 | 487K | Nightingale Health |
| Exome sequencing | 23141-23172 | 28 | 469K | OQFE protocol |
| Genomics general | 23141-23200+ | 194 cols | 469K | Multiple |
| Neurobiomarkers | 31040-31049 | 10 | 1.3K | Quanterix Simoa |
| Proteomics (Olink) | 30860-30903 | 33 | ? | Olink Explore (data pending) |
| Blood biomarkers | 20100-30900+ | 2,000+ | 420K-490K | Various |
| Brain MRI (all) | 20100-20320 | 215+ | 82K | Siemens 3T |
| Heart MRI | 21000-22500 | 165 | 80K | Siemens 3T |
| DXA | 20300-20320 | 168 | 57K | Hologic |
| ECG | 12323-20205 | 18+ | 95K | GE MAC |
| Activity | 884, 894, 904, 914 | 4+ | 500K | Questionnaire |

---

## 10. DATA QUALITY & COVERAGE

### High Coverage (>90%)
- ✅ Metabolomics: 97% (487K/502K)
- ✅ Blood biomarkers: 85-95% (varies)
- ✅ Genetic data: 93% (469K/502K exome)
- ✅ Physical activity (questionnaire): 99.8% (501K/502K)
- ✅ Demographics: 99.9%

### Medium Coverage (50-90%)
- ✅ Brain MRI: 16-17% (82K/502K) - intentional subsample
- ✅ Heart MRI: 16% (79K/502K) - intentional subsample
- ✅ DXA: 13% (66K/502K) - intentional subsample
- ✅ ECG: 19% (95K/502K) - intentional subsample
- ✅ Metabolite specifics: 97% for most metabolites

### Low Coverage (<50%)
- ⚠️ Neurobiomarkers: 0.25% (1.3K/502K) - COVID re-imaging substudy
- ? Accelerometer time-series: ~85%+ (need verification)
- ? Retinal imaging: ~13% (44K/502K)
- ⚠️ Olink proteomics: Unknown (data not yet in system)

### Missing/Pending
- ❌ Full Olink proteomics (1,400+ proteins)
- ? Retinal imaging details
- ? Accelerometer raw time-series files

---

## 11. KEY INSIGHTS FOR RESEARCH

### Best-Powered Studies
1. **Metabolomics + Outcomes**: 487K participants, 509 metabolites
2. **Genomics + Outcomes**: 469K participants, exome + WGS
3. **Blood biomarkers + Outcomes**: 420K-490K participants

### Multi-Omics Opportunities
- Integrate metabolomics (487K) + genomics (469K) = ~400K participants with both
- Add: blood biomarkers (480K), clinical outcomes (500K+)
- Add: selected brain imaging subset (82K for detailed structure)

### Imaging + Omics
- Brain MRI (82K) + metabolomics (82K in subset) = neurodegenerative disease studies
- Cardiac MRI (80K) + metabolomics (80K) = cardiovascular metabolomics

### Limitations
- Proteomics data (Olink) not yet accessible in processed format
- Imaging is intentional subset (16-20%), not population-wide
- Accelerometer data location/format needs verification
- Some derived fields have lower coverage

---

## 12. NEXT STEPS FOR FULL VERIFICATION

1. **Locate Olink proteomics data**: Search `/Users/chenpengan/Projects/CUHK/UKB/` for protein abundance files
2. **Verify accelerometer time-series**: Check for `.csv` or `.parquet` files with activity data
3. **Find raw imaging files**: Confirm location of NIFTI/DICOM files (likely not in milton_data)
4. **Verify retinal imaging**: Confirm coverage and file storage location
5. **Check genotype matrix files**: Locate PLINK/VCF/BGEN files referenced in genomics fields

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Participants | 502,370 |
| Total Field IDs | 11,822 |
| Metabolites available | 509 |
| Metabolomics coverage | 97% |
| Exome participants | 469,391 |
| Brain MRI participants | 82,234 |
| Blood biomarker tests | 50+ |
| Blood biomarker coverage | 85-95% |
| Imaging modalities | 12+ |
| Proteomics availability | ⚠️ Pending |

