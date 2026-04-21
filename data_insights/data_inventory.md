# UK Biobank Data Inventory

## Data Sources

### Processed Parquet (Fast Path)
**Location:** `/Users/chenpengan/Projects/CUHK/milton_data/`

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `ukb.parquet/` (4 parts) | 502,370 | 2,031 (484 field IDs) | 363 MB | Main biomarker + phenotype matrix |
| `hesin_diag.parquet/` | 6,946,795 | 4 | 37 MB | ICD10/ICD9 diagnosis records (tall format) |
| `death_cause.parquet/` | 112,917 | 5 | 576 KB | Cause-of-death records (tall format) |

### Raw CSVs (Comprehensive)
**Location:** `/Users/chenpengan/Projects/CUHK/UKB/`

| File | Columns | Field IDs | Size |
|------|---------|-----------|------|
| `UKB/ukb672073.csv` | 30,799 | 8,868 | 54 GB |
| `UKB/ukb671626.csv` | 18,506 | 3,461 | 35 GB |
| `UKB_info/ukb672073_Biological_Samples.csv` | 1,775 | 874 | 3.5 GB |
| `UKB_info/ukb672073_Health_Related_Outcomes.csv` | 4,896 | 2,564 | 7.3 GB |
| `UKB_info/ukb672073_Online_Follow_up.csv` | 5,467 | 1,106 | 8.3 GB |
| `UKB_info/ukb672073_Genomics.csv` | 194 | 131 | 1.0 GB |
| `UKB_info/ukb672073_Additional_Exposures.csv` | 279 | 245 | 679 MB |
| `UKB_info/ukb672073_Population_Characteristics.csv` | 34 | 33 | 78 MB |

### Metadata
| File | Content | Size |
|------|---------|------|
| `field.txt` | 11,821 field definitions | 4 MB |
| `category.txt` | 410 categories | 172 KB |
| `esimpint.txt` | Simple integer encodings | 649 KB |
| `ehierint.txt` | Hierarchical integer encodings | 1.4 MB |
| `Data_Dictionary_Showcase.csv` | Curated field documentation | 3.9 MB |

## Coverage Gap

Parquet covers **484 / 4,953** field IDs (10%) from category CSVs.

| Category | Total Fields | In Parquet | Missing |
|----------|-------------|------------|---------|
| Population_Characteristics | 33 | 4 | 29 |
| Biological_Samples | 874 | 437 | 437 |
| Health_Related_Outcomes | 2,564 | 14 | 2,550 |
| Additional_Exposures | 245 | 5 | 240 |
| Online_Follow_up | 1,106 | 6 | 1,100 |
| Genomics | 131 | 0 | 131 |

## Parquet Field Categories

### Blood Biochemistry (30 fields: 30600–30900)
Albumin, Alkaline phosphatase, ALT, ApoA, ApoB, AST, Bilirubin, Urea, Calcium, Cholesterol, Creatinine, CRP, Cystatin C, GGT, Glucose, HbA1c, HDL, IGF-1, LDL, Lipoprotein A, Oestradiol, Phosphate, SHBG, Testosterone, Total protein, Triglycerides, Urate, Vitamin D

### Blood Count (31 fields: 30000–30300)
WBC, RBC, Haemoglobin, Haematocrit, MCV, MCH, MCHC, RDW, Platelets, Lymphocytes, Monocytes, Neutrophils, Eosinophils, Basophils, Reticulocytes

### Demographics (6 fields)
Sex (31), Year of birth (34), Month of birth (52), Ethnic background (21000), Age (21003/21022)

### Anthropometric (5 fields)
Waist circumference (48), Hip circumference (49), Standing height (50), BMI (21001/23104)

### Blood Pressure (5 fields)
Systolic/Diastolic automated (4079/4080), manual (93/94), Pulse rate (102)

### ICD10 Diagnoses (wide format: 41270, ~259 columns)
Retained for compatibility; prefer `hesin_diag.parquet` for queries.

## Diagnosis Records (hesin_diag)

Top 10 ICD10 codes by patient count:
1. **I10** Essential hypertension — 159,860 (31.8%)
2. **Z86** Personal history — 128,975 (25.7%)
3. **Z92** Personal history of treatment — 90,965 (18.1%)
4. **E78** Lipoprotein disorders — 82,940 (16.5%)
5. **Z87** Personal history — 69,770 (13.9%)
6. **M54** Dorsalgia — 62,741 (12.5%)
7. **K57** Diverticular disease — 61,424 (12.2%)
8. **J06** Upper respiratory infections — 50,424 (10.0%)
9. **I25** Chronic ischaemic heart disease — 47,089 (9.4%)
10. **E11** Type 2 diabetes — 43,769 (8.7%)

## Death Records (death_cause)

- 44,198 deceased participants (8.8% mortality)
- 112,917 total cause-of-death records
- Top causes: I259 (heart disease), C349 (lung cancer), J189 (pneumonia)
