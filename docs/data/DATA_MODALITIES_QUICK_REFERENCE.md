# UK BIOBANK - DATA MODALITIES QUICK REFERENCE

**Last Updated**: 2026-04-23  
**Total Participants**: 502,370  
**Data Location**: `/Users/chenpengan/Projects/CUHK/milton_data/`

---

## ✅ CONFIRMED OMICS & IMAGING DATA ON DISK

### 1. METABOLOMICS (NMR) ✅ READY TO USE
- **Coverage**: 97% (487K/502K participants)
- **Format**: 509 metabolites in columns
- **Location**: `Biological_Samples.parquet` (Category 220-222)
- **Field IDs**: 20280-23948 (~1,712 in catalog)
- **Platform**: Nightingale Health 1H NMR
- **Types**: Amino acids, lipoproteins, fatty acids, glucose markers
- **Size**: 207 MB
- **Status**: ✅ IMMEDIATELY USABLE

### 2. GENOMICS ✅ READY TO USE (with caveats)
- **Coverage**: 93% exome (469K/502K), WGS varies
- **Format**: Field references to VCF/CRAM/PLINK files
- **Location**: `Genomics.parquet` (194 columns)
- **Field IDs**: 23141-23172 (28 catalog fields)
- **Size**: 371 MB
- **Status**: ✅ METADATA READY, actual genotype files location TBD

### 3. BLOOD BIOMARKERS ✅ READY TO USE
- **Coverage**: 85-95% (varies by test)
- **Format**: Numeric measurements in columns
- **Location**: `Biological_Samples.parquet`
- **Types**: Blood counts, biochemistry, lipids, vitamins
- **Tests**: 50+ routine clinical tests
- **Participants**: 420K-490K (varies)
- **Status**: ✅ IMMEDIATELY USABLE

### 4. BRAIN IMAGING ✅ READY TO USE
- **Coverage**: 16-17% intentional subsample (82K/502K)
- **Format**: NIFTI/processed derivatives
- **Modalities**:
  - T1 structural (20 fields, 82K)
  - T2/FLAIR (5 fields, 80K)
  - Diffusion dMRI (691 fields, 79K)
  - Resting fMRI (45 fields, 82K)
  - Task fMRI (26 fields, 67K)
  - Susceptibility (36 fields, 74K)
  - Brain atlases (1,000+ fields, 40-83K)
- **Status**: ✅ METADATA/DERIVATIVES AVAILABLE

### 5. CARDIAC IMAGING ✅ READY TO USE
- **Coverage**: 16-20% (79K-95K/502K)
- **Types**:
  - Heart MRI (10 fields, 80K)
  - Cardiac function derived (155 fields, varies)
  - ECG (18 fields, 95K)
  - Pulse wave analysis (23 fields, 80K)
- **Status**: ✅ METADATA AVAILABLE

### 6. BONE DENSITY (DXA) ✅ READY TO USE
- **Coverage**: 13% (66K/502K)
- **Format**: Bone composition indices
- **Fields**: 168 total (body comp + bone density)
- **Coverage**: Hip, spine, forearm, whole body
- **Status**: ✅ METADATA AVAILABLE

### 7. ABDOMINAL IMAGING ✅ READY TO USE
- **Coverage**: 13-40% (varies)
- **Types**: Liver MRI, pancreas, kidney MRI, organ composition
- **Derived pipelines**: AMRA, Calico, Uppsala
- **Status**: ✅ METADATA AVAILABLE

### 8. RETINAL IMAGING ✅ REFERENCED
- **Coverage**: 13% (44K/502K)
- **Types**: Fundus images, optic disc, macular thickness
- **Fields**: 80+ derived fields
- **Status**: ⚠️ Need to verify raw image location

### 9. ACTIVITY & ACCELEROMETER ✅ PARTIAL
- **Self-reported**: ✅ 500K+ participants
  - Physical activity types, frequency, duration (Fields 884, 894, 904, 914)
- **Wrist accelerometer**: ⚠️ Data location needs verification
  - Expected ~85%+ coverage
  - Time-series activity counts
  - Field IDs: likely 90002-90100 range

### 10. NEUROBIOMARKERS ✅ LIMITED
- **Coverage**: 0.25% (1.3K/502K - COVID re-imaging subsample)
- **Biomarkers**: Amyloid-β40, Amyloid-β42, GFAP, NfL, pTau-181
- **Platform**: Quanterix Simoa HD-X
- **Field IDs**: 31040-31049
- **Status**: ✅ AVAILABLE (small subset)

---

## ⚠️ PENDING/UNCLEAR

### 1. PROTEOMICS (OLINK) ⚠️ DATA PENDING
- **Status**: Field definitions exist (30860-30903) but **data structure is EMPTY**
- **Expected**: 1,400+ proteins from Olink Explore 1536
- **Expected coverage**: ~53,000 participants
- **File**: `/Users/chenpengan/Projects/CUHK/milton_data/olink.parquet/`
- **Issue**: Only contains 'eid' column - actual protein data not loaded
- **Action**: Locate raw proteomics file or secondary data store

### 2. ACCELEROMETER TIME-SERIES ⚠️ LOCATION UNCLEAR
- **Status**: Self-reported activity confirmed, raw time-series location unknown
- **Expected format**: .csv or .parquet with minute/hour-level activity counts
- **Coverage**: Estimated 85%+
- **Action**: Search `/Users/chenpengan/Projects/CUHK/UKB/` for activity files

### 3. RAW IMAGING FILES ⚠️ NOT IN PARQUET
- **Status**: Metadata/derivatives in parquet, but raw NIFTI/DICOM location TBD
- **Types**: Brain MRI, Heart MRI, abdominal imaging, DXA, retinal images
- **Action**: Likely in `/Users/chenpengan/Projects/CUHK/UKB/` subdirectories

### 4. GENOTYPE MATRICES ⚠️ FILE REFERENCES ONLY
- **Status**: Field IDs reference PLINK/VCF/BGEN but actual files location TBD
- **Coverage**: 469K exome, variable WGS
- **Format**: OQFE protocol (improved over older SPB/FE)
- **Action**: Locate genotype files in raw data directory

---

## 🎯 BEST USE CASES (IMMEDIATELY AVAILABLE)

### High-Power Studies
1. **Metabolomics biomarker discovery**: 487K × 509 metabolites ✅
2. **Blood biomarker + outcomes**: 480K × 50+ tests ✅
3. **Multi-omics integration**: ~400K with both metabolomics + genomics (when genomics located)

### Imaging + Biomarker Studies
1. **Brain MRI + metabolomics**: 82K with detailed imaging ✅
2. **Cardiac MRI + biomarkers**: 80K with heart imaging ✅
3. **DXA + blood biomarkers**: 66K with bone + biochemistry ✅

### Genetic Studies
1. **Exome-phenotype associations**: 469K participants ✅
2. **PRS validation**: 485K participants ✅
3. **Multi-biomarker GWAS**: 400K+ with metabolomics ✅

---

## 📊 COVERAGE MATRIX

| Data Type | Participants | % Coverage | Status | Notes |
|-----------|--------------|-----------|--------|-------|
| Demographics | 502K | 100% | ✅ | All fields |
| Blood biomarkers | 480K | 95% | ✅ | Routine tests |
| Metabolomics | 487K | 97% | ✅ | NMR platform |
| Exome | 469K | 93% | ✅ | OQFE protocol |
| Brain MRI | 82K | 16% | ✅ | Intentional |
| Heart MRI | 80K | 16% | ✅ | Intentional |
| DXA | 66K | 13% | ✅ | Intentional |
| ECG | 95K | 19% | ✅ | Intentional |
| Retinal | 44K | 9% | ✅ | Fundus photos |
| Neurobiomarkers | 1.3K | 0.3% | ✅ | COVID subset |
| Proteomics (Olink) | ? | ? | ⚠️ | Data pending |
| WGS | 150-300K | 30-60% | ✅ | Varies by release |
| Accelerometer | 450K+ | 85%+ | ⚠️ | Location TBD |

---

## 📁 KEY FILES & LOCATIONS

```
/Users/chenpengan/Projects/CUHK/milton_data/
├── categories/
│   ├── Biological_Samples.parquet          ⭐ Metabolomics + blood
│   ├── Genomics.parquet                    ⭐ Exome + WGS metadata
│   ├── Health_Related_Outcomes.parquet     ⭐ Outcomes/diagnoses
│   └── [others...]
├── field.txt                               Field catalog (11,822 entries)
├── category.txt                            Category definitions
├── olink.parquet/                          ⚠️ Empty (proteomics pending)
└── [other supporting files]

/Users/chenpengan/Projects/CUHK/UKB/
├── UKB/                                    Raw data (location TBD)
├── UKB_info/                               Documentation
└── [imaging/genetic data subdirs?]
```

---

## 🔍 QUICK FIELD ID LOOKUP

| Data Type | Field ID Range | Count | Usage |
|-----------|-----------------|-------|-------|
| Metabolites | 20280-23948 | 1,712 | Column selection |
| Exome refs | 23141-23172 | 28 | File references |
| Proteins (routine) | 30860-30903 | 33 | Blood chemistry |
| Brain MRI | 20100+ | 215+ | Image derivatives |
| Blood counts | Various | 100+ | Hematology |
| Biochemistry | 20000-30000 | 500+ | Clinical tests |
| Neurobiomarkers | 31040-31049 | 10 | Plasma Aβ, tau |

---

## ✅ ACTION CHECKLIST

- [x] Metabolomics: Ready to use
- [x] Blood biomarkers: Ready to use
- [x] Genomics metadata: Ready (actual genotypes need location)
- [x] Brain imaging metadata: Ready (images need location)
- [x] Cardiac imaging: Ready (images need location)
- [ ] Olink proteomics: Locate raw data or reload
- [ ] Accelerometer time-series: Locate raw files
- [ ] Raw imaging files: Confirm directory structure
- [ ] Retinal imaging: Verify coverage and location
- [ ] Full genotype matrices: Locate PLINK/VCF/BGEN files

---

**For detailed field-level information, see**: [`UKB_DATA_FIELDS_INVENTORY.md`](UKB_DATA_FIELDS_INVENTORY.md)

