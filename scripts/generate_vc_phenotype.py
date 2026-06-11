"""Generate vc_samples.parquet from the 28-sample WGS cohort metadata."""

from pathlib import Path

import pandas as pd

SAMPLES = [
    {"donor": "S1", "part": "Occipital", "sample_id": "S1-58Y-F-1", "age": 58, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S1-58Y-F-1.genotyper.vcf.gz"},
    {"donor": "J1", "part": "Temple", "sample_id": "J1-41Y-F", "age": 41, "sex": "F", "phenotype": "Juvenile_White", "vcf_file": "J1-41Y-F.genotyper.vcf.gz"},
    {"donor": "J2", "part": "Occipital", "sample_id": "J2-35Y-F", "age": 35, "sex": "F", "phenotype": "Juvenile_White", "vcf_file": "J2-35Y-F.genotyper.vcf.gz"},
    {"donor": "J3", "part": "Occipital_and_Vertex", "sample_id": "J3-27Y-M", "age": 27, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J3-27Y-M.genotyper.vcf.gz"},
    {"donor": "J4", "part": "Occipital", "sample_id": "J4-19Y-M", "age": 19, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J4-19Y-M.genotyper.vcf.gz"},
    {"donor": "J5", "part": "Vertex", "sample_id": "J5-23Y-M", "age": 23, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J5-23Y-M.genotyper.vcf.gz"},
    {"donor": "J6", "part": "Vertex", "sample_id": "J6-25Y-F", "age": 25, "sex": "F", "phenotype": "Juvenile_White", "vcf_file": "J6-25Y-F.genotyper.vcf.gz"},
    {"donor": "J7", "part": "Frontal", "sample_id": "J7-34Y-F", "age": 34, "sex": "F", "phenotype": "Juvenile_White", "vcf_file": "J7-34Y-F.genotyper.vcf.gz"},
    {"donor": "J8", "part": "Temple", "sample_id": "J8-31Y-M", "age": 31, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J8-31Y-M.genotyper.vcf.gz"},
    {"donor": "J9", "part": "Occipital", "sample_id": "J9-21Y-M", "age": 21, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J9-21Y-M.genotyper.vcf.gz"},
    {"donor": "S2", "part": "Occipital", "sample_id": "S2-44Y-M", "age": 44, "sex": "M", "phenotype": "Senile_White", "vcf_file": "S2-44Y-M.genotyper.vcf.gz"},
    {"donor": "S3", "part": "Temple", "sample_id": "S3-48Y-F", "age": 48, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S3-48Y-F.genotyper.vcf.gz"},
    {"donor": "J10", "part": "Vertex", "sample_id": "J10-28Y-M", "age": 28, "sex": "M", "phenotype": "Juvenile_White", "vcf_file": "J10-28Y-M.genotyper.vcf.gz"},
    {"donor": "S4", "part": "Temple", "sample_id": "S4-47Y-F", "age": 47, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S4-47Y-F.genotyper.vcf.gz"},
    {"donor": "S6", "part": "Temple", "sample_id": "S6-56Y-M", "age": 56, "sex": "M", "phenotype": "Senile_White", "vcf_file": "S6-56Y-M.genotyper.vcf.gz"},
    {"donor": "S7", "part": "Temple", "sample_id": "S7-56Y-M", "age": 56, "sex": "M", "phenotype": "Senile_White", "vcf_file": "S7-56Y-M.genotyper.vcf.gz"},
    {"donor": "S8", "part": "Occipital", "sample_id": "S8-75Y-F", "age": 75, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S8-75Y-F.genotyper.vcf.gz"},
    {"donor": "V1", "part": "Occipital", "sample_id": "V1-55Y-M", "age": 55, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V1-55Y-M.genotyper.vcf.gz"},
    {"donor": "S9", "part": "Occipital", "sample_id": "S9-66Y-M", "age": 66, "sex": "M", "phenotype": "Senile_White", "vcf_file": "S9-66Y-M.genotyper.vcf.gz"},
    {"donor": "S10", "part": "Occipital", "sample_id": "S10-63Y-F", "age": 63, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S10-63Y-F.genotyper.vcf.gz"},
    {"donor": "S11", "part": "Occipital", "sample_id": "S11-61Y-F", "age": 61, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S11-61Y-F.genotyper.vcf.gz"},
    {"donor": "V2", "part": "Occipital", "sample_id": "V2-58Y-M", "age": 58, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V2-58Y-M.genotyper.vcf.gz"},
    {"donor": "S12", "part": "Occipital", "sample_id": "S12-70Y-F", "age": 70, "sex": "F", "phenotype": "Senile_White", "vcf_file": "S12-70Y-F.genotyper.vcf.gz"},
    {"donor": "S13", "part": "Occipital", "sample_id": "S13-69Y-M", "age": 69, "sex": "M", "phenotype": "Senile_White", "vcf_file": "S13-69Y-M.genotyper.vcf.gz"},
    {"donor": "V3", "part": "Occipital", "sample_id": "V3-31Y-M", "age": 31, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V3-31Y-M.genotyper.vcf.gz"},
    {"donor": "V4", "part": "Occipital", "sample_id": "V4-48Y-M", "age": 48, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V4-48Y-M.genotyper.vcf.gz"},
    {"donor": "V5", "part": "Vertex", "sample_id": "V5-24Y-M", "age": 24, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V5-24Y-M.genotyper.vcf.gz"},
    {"donor": "V6", "part": "Vertex", "sample_id": "V6-12Y-M", "age": 12, "sex": "M", "phenotype": "Vitiligo_White", "vcf_file": "V6-12Y-M.genotyper.vcf.gz"},
]

BW_VCF_DIR = Path("/Files/ResultData/BW_WGS_vcf")
VC_VCF_DIR = Path("/work/c-chenpengan/biobank_agent/data/vc_wgs_vcf")
OUT_DIR = Path("/work/c-chenpengan/biobank_agent/data/virtualcell")


def main() -> None:
    rows = []
    for s in SAMPLES:
        bw_path = BW_VCF_DIR / s["vcf_file"]
        vc_path = VC_VCF_DIR / s["vcf_file"]
        vcf_path = str(bw_path) if bw_path.exists() else str(vc_path) if vc_path.exists() else ""
        phenotype_group = s["donor"][0]  # J / S / V
        rows.append({
            "sample_id": s["sample_id"],
            "donor": s["donor"],
            "part": s["part"],
            "age": s["age"],
            "sex": s["sex"],
            "is_male": 1 if s["sex"] == "M" else 0,
            "phenotype": s["phenotype"],
            "phenotype_group": phenotype_group,
            "vcf_path": vcf_path,
        })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "vc_samples.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Written {len(df)} samples to {out_path}")
    print(f"\nPhenotype distribution:")
    print(df["phenotype"].value_counts().to_string())
    print(f"\nSex distribution:")
    print(df["sex"].value_counts().to_string())
    print(f"\nAge range: {df['age'].min()}-{df['age'].max()}")

    _generate_field_catalog(df)


FIELD_DESCRIPTIONS = {
    "sample_id": "Sample ID",
    "donor": "Donor ID",
    "part": "Sampling site (scalp region)",
    "age": "Age at sampling (years)",
    "sex": "Biological sex (M/F)",
    "is_male": "Is male (0/1 coded)",
    "phenotype": "Hair phenotype (Juvenile_White, Senile_White, Vitiligo_White)",
    "phenotype_group": "Phenotype group code (J/S/V)",
    "vcf_path": "Path to genotyper VCF file",
}

FIELD_CATEGORIES = {
    "demographics": ("Demographics", ["sample_id", "donor", "age", "sex", "is_male"]),
    "phenotype": ("Phenotype", ["phenotype", "phenotype_group"]),
    "sampling": ("Sampling", ["part"]),
    "genomics": ("Genomics", ["vcf_path"]),
}


def _generate_field_catalog(df: pd.DataFrame) -> None:
    field_path = OUT_DIR / "field.txt"
    cat_path = OUT_DIR / "category.txt"

    with open(cat_path, "w") as f:
        f.write("category_id\ttitle\tavailability\n")
        for cat_id, (title, _) in FIELD_CATEGORIES.items():
            f.write(f"{cat_id}\t{title}\tAvailable\n")

    with open(field_path, "w") as f:
        f.write("field_id\ttitle\n")
        for col in df.columns:
            title = FIELD_DESCRIPTIONS.get(col, col)
            f.write(f"{col}\t{title}\n")

    print(f"\nGenerated field catalogue: {field_path}")
    print(f"Generated category catalogue: {cat_path}")


if __name__ == "__main__":
    main()
