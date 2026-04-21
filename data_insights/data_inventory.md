# UK Biobank Data Inventory

## Data Architecture

DuckDB serves as the unified query layer, reading from two tiers:
- **Tier 1 (fast):** Pre-built parquet files — zero-copy column scans
- **Tier 2 (fallback):** Raw CSV files — lazy-loaded when a field isn't in parquet

## Parquet Datasets

### Core Biomarkers (`ukb.parquet/`)

| Property | Value |
|----------|-------|
| Rows | 502,370 |
| Columns | 2,031 (484 unique field IDs) |
| Size | 363 MB (4 SNAPPY-compressed partitions) |
| Format | Row-partitioned, `{field_id}-{instance}.{array}` column names |

### Category Parquets (`categories/`)

Built from raw CSVs via `bb rebuild-parquet`. Each category is a separate file.

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

### Clinical Records

| Dataset | Rows | Columns | Size |
|---------|------|---------|------|
| `hesin_diag.parquet/` (ICD10 diagnoses) | 6,946,795 | 4 | 37 MB |
| `death_cause.parquet/` (cause of death) | 112,917 | 5 | 576 KB |

## Raw CSV Sources

All under `$UKB_RAW_DIR/UKB_info/`:

| File | Columns | Size |
|------|---------|------|
| `ukb672073_Population_Characteristics.csv` | 34 | 78 MB |
| `ukb672073_Biological_Samples.csv` | 1,775 | 3.5 GB |
| `ukb672073_Additional_Exposures.csv` | 279 | 679 MB |
| `ukb672073_Genomics.csv` | 194 | 1.0 GB |
| `ukb672073_Health_Related_Outcomes.csv` | 4,896 | 7.3 GB |
| `ukb672073_Online_Follow_up.csv` | 5,467 | 8.3 GB |

Main dataset (`$UKB_RAW_DIR/UKB/ukb672073.csv`): 30,799 columns, 54 GB.

## Metadata

| File | Records | Size | Description |
|------|---------|------|-------------|
| `field.txt` | 11,821 | 4 MB | Field ID → name, category, value type |
| `category.txt` | 410 | 172 KB | Category ID → name |
| `esimpint.txt` | 18,079 | 649 KB | Simple integer encoding lookup |
| `ehierint.txt` | 28,901 | 1.4 MB | Hierarchical encoding lookup |

## Key Biomarker Groups

### Blood Biochemistry (30 fields: 30600–30900)
Albumin, ALT, AST, ApoA, ApoB, Bilirubin, Calcium, Cholesterol, Creatinine, CRP, Cystatin C, GGT, Glucose, HbA1c, HDL, IGF-1, LDL, Lipoprotein A, Phosphate, SHBG, Testosterone, Total protein, Triglycerides, Urate, Urea, Vitamin D

### Blood Count (31 fields: 30000–30300)
WBC, RBC, Haemoglobin, Haematocrit, MCV, MCH, MCHC, RDW, Platelets, Lymphocytes, Monocytes, Neutrophils, Eosinophils, Basophils, Reticulocytes

### Demographics (6 fields)
Sex (31), Year of birth (34), Month of birth (52), Ethnic background (21000), Age at recruitment (21003/21022)

### Anthropometric (5 fields)
Waist circumference (48), Hip circumference (49), Standing height (50), BMI (21001/23104)

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

## Field Routing

When the agent queries a field, `DataManager.field_source()` routes to the fastest source:

1. **`biomarkers` view** — core parquet (484 fields, fastest)
2. **`cat_*` views** — category parquets (4,487 fields, fast)
3. **`csv_*` views** — raw CSV fallback (lazy-loaded, slow but comprehensive)
