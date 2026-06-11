"""VirtualCell/BWhair multimodal manifests and lightweight readers.

The VirtualCell benchmark data are intentionally not a conventional UKB table:
WGS is one blood sample per donor, while Stereo-seq/scRNA/scATAC are h5ad
modalities keyed through donor/sample metadata. This module keeps that domain
knowledge in the data layer so prompts and tests do not need to spell it out.
"""

from __future__ import annotations

import importlib.util
import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_WGS_MANIFEST_TSV = """Donor\tPart\tsampleID\tAge\tSex\tPhenotype\tfilePath
S1\tOccipital\tS1-58Y-F-1\t58Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S1-58Y-F-1.genotyper.vcf.gz
J1\tTemple\tJ1-41Y-F\t41Y\tFemale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J1-41Y-F.genotyper.vcf.gz
J2\tOccipital\tJ2-35Y-F\t35Y\tFemale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J2-35Y-F.genotyper.vcf.gz
J3\tOccipital_and_Vertex\tJ3-27Y-M\t27Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J3-27Y-M.genotyper.vcf.gz
J4\tOccipital\tJ4-19Y-M\t19Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J4-19Y-M.genotyper.vcf.gz
J5\tVertex\tJ5-23Y-M\t23Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J5-23Y-M.genotyper.vcf.gz
J6\tVertex\tJ6-25Y-F\t25Y\tFemale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J6-25Y-F.genotyper.vcf.gz
J7\tFrontal\tJ7-34Y-F\t34Y\tFemale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J7-34Y-F.genotyper.vcf.gz
J8\tTemple\tJ8-31Y-M\t31Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J8-31Y-M.genotyper.vcf.gz
J9\tOccipital\tJ9-21Y-M\t21Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J9-21Y-M.genotyper.vcf.gz
S2\tOccipital\tS2-44Y-M\t44Y\tMale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S2-44Y-M.genotyper.vcf.gz
S3\tTemple\tS3-48Y-F\t48Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S3-48Y-F.genotyper.vcf.gz
J10\tVertex\tJ10-28Y-M\t28Y\tMale\tJuvenile_White\t/Files/ResultData/BW_WGS_vcf/J10-28Y-M.genotyper.vcf.gz
S4\tTemple\tS4-47Y-F\t47Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S4-47Y-F.genotyper.vcf.gz
S6\tTemple\tS6-56Y-M\t56Y\tMale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S6-56Y-M.genotyper.vcf.gz
S7\tTemple\tS7-56Y-M\t56Y\tMale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S7-56Y-M.genotyper.vcf.gz
S8\tOccipital\tS8-75Y-F\t75Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S8-75Y-F.genotyper.vcf.gz
V1\tOccipital\tV1-55Y-M\t55Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V1-55Y-M.genotyper.vcf.gz
S9\tOccipital\tS9-66Y-M\t66Y\tMale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S9-66Y-M.genotyper.vcf.gz
S10\tOccipital\tS10-63Y-F\t63Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S10-63Y-F.genotyper.vcf.gz
S11\tOccipital\tS11-61Y-F\t61Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S11-61Y-F.genotyper.vcf.gz
V2\tOccipital\tV2-58Y-M\t58Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V2-58Y-M.genotyper.vcf.gz
S12\tOccipital\tS12-70Y-F\t70Y\tFemale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S12-70Y-F.genotyper.vcf.gz
S13\tOccipital\tS13-69Y-M\t69Y\tMale\tSenile_White\t/Files/ResultData/BW_WGS_vcf/S13-69Y-M.genotyper.vcf.gz
V3\tOccipital\tV3-31Y-M\t31Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V3-31Y-M.genotyper.vcf.gz
V4\tOccipital\tV4-48Y-M\t48Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V4-48Y-M.genotyper.vcf.gz
V5\tVertex\tV5-24Y-M\t24Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V5-24Y-M.genotyper.vcf.gz
V6\tVertex\tV6-12Y-M\t12Y\tMale\tVitiligo_White\t/Files/ResultData/BW_WGS_vcf/V6-12Y-M.genotyper.vcf.gz
"""


DEFAULT_BWHAIR_MANIFEST_TSV = """Filename\tTechnology\tPath\tSize\tCells\tDescription
J10-WB-28Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t315.6 MB\t47104\tBWhair spatial dataset
S12-WB-70Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t378.2 MB\t78812\tBWhair spatial dataset
J1-B-41Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t283.5 MB\t49727\tBWhair spatial dataset
S14-B-70Y-M-1_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t328.9 MB\t45151\tBWhair spatial dataset
S10-WB-63Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t340.2 MB\t67371\tBWhair spatial dataset
J8-WB-31Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t404.1 MB\t81176\tBWhair spatial dataset
S7-WB-56Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t220.9 MB\t43615\tBWhair spatial dataset
J10-B-28Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t663.5 MB\t80242\tBWhair spatial dataset
V3-WB-31Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t336.8 MB\t59736\tBWhair spatial dataset
S2-B-44Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t380.1 MB\t74027\tBWhair spatial dataset
J10-W-28Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t693.7 MB\t98997\tBWhair spatial dataset
S11-B-61Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t843.2 MB\t151371\tBWhair spatial dataset
J5-B-23Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t346.6 MB\t64973\tBWhair spatial dataset
S7-B-56Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t351.6 MB\t49643\tBWhair spatial dataset
S10-B-63Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t655.1 MB\t102804\tBWhair spatial dataset
V1-B-55Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t520.6 MB\t100512\tBWhair spatial dataset
S4-W-47Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t435.3 MB\t89185\tBWhair spatial dataset
J3-G-27Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t340.1 MB\t64192\tBWhair spatial dataset
S11-W-61Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t399.9 MB\t68665\tBWhair spatial dataset
S14-W-70Y-M-2_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t163.4 MB\t27297\tBWhair spatial dataset
S2-WB-44Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t371.5 MB\t113573\tBWhair spatial dataset
J6-B-25Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t225.6 MB\t62445\tBWhair spatial dataset
J2-W-35Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t298.7 MB\t46130\tBWhair spatial dataset
V5-WB-24Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t289.0 MB\t66967\tBWhair spatial dataset
J9-W-21Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t325.3 MB\t71150\tBWhair spatial dataset
S13-WB-69Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t440.2 MB\t81273\tBWhair spatial dataset
J1-G-41Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t190.5 MB\t49268\tBWhair spatial dataset
J4-G-19Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t322.4 MB\t55138\tBWhair spatial dataset
S8-WB-75Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t445.8 MB\t82662\tBWhair spatial dataset
J9-B-21Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t486.5 MB\t96602\tBWhair spatial dataset
S12-B-70Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t802.7 MB\t103026\tBWhair spatial dataset
J9-WB-21Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t319.7 MB\t72985\tBWhair spatial dataset
J6-W-25Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t262.6 MB\t62357\tBWhair spatial dataset
V4-W-48Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t551.4 MB\t100231\tBWhair spatial dataset
V2-B-58Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t343.5 MB\t49096\tBWhair spatial dataset
V1-W-55Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t535.7 MB\t82067\tBWhair spatial dataset
J3-B-27Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t410.0 MB\t94245\tBWhair spatial dataset
S4-WB-47Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t431.9 MB\t84749\tBWhair spatial dataset
S8-B-75Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t471.5 MB\t88063\tBWhair spatial dataset
J2-B-35Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t453.9 MB\t72014\tBWhair spatial dataset
S12-W-70Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t689.8 MB\t94222\tBWhair spatial dataset
J6-GB-25Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t214.4 MB\t58158\tBWhair spatial dataset
S9-WB-66Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t466.2 MB\t74596\tBWhair spatial dataset
V5-B-24Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t545.7 MB\t133273\tBWhair spatial dataset
S14-B-70Y-M-3_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t269.7 MB\t62267\tBWhair spatial dataset
J5-WB-23Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t370.1 MB\t77268\tBWhair spatial dataset
S9-W-66Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t708.9 MB\t122224\tBWhair spatial dataset
V5-W-24Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t570.1 MB\t120799\tBWhair spatial dataset
V1-WB-55Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t379.5 MB\t56248\tBWhair spatial dataset
S8-W-75Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t387.3 MB\t64799\tBWhair spatial dataset
S7-W-56Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t513.9 MB\t87951\tBWhair spatial dataset
J7-B-34Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t287.9 MB\t63038\tBWhair spatial dataset
V4-B-48Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t581.6 MB\t141598\tBWhair spatial dataset
S6-WB-56Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t264.9 MB\t68242\tBWhair spatial dataset
S13-W-69Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t359.8 MB\t71871\tBWhair spatial dataset
S4-B-47Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t512.1 MB\t113728\tBWhair spatial dataset
S9-B-66Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t510.5 MB\t104623\tBWhair spatial dataset
V6-W-12Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t559.4 MB\t88699\tBWhair spatial dataset
V6-B-12Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t648.3 MB\t90847\tBWhair spatial dataset
J4-B-19Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t206.1 MB\t54857\tBWhair spatial dataset
J3-W-27Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t379.0 MB\t82468\tBWhair spatial dataset
J5-G-23Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t151.4 MB\t28791\tBWhair spatial dataset
S14-W-70Y-M-1_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t123.6 MB\t21471\tBWhair spatial dataset
V3-W-31Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t558.1 MB\t112796\tBWhair spatial dataset
V3-B-31Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t579.4 MB\t113026\tBWhair spatial dataset
S2-W-44Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t485.9 MB\t89353\tBWhair spatial dataset
S5-WB-45Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t340.6 MB\t79172\tBWhair spatial dataset
S11-WB-61Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t558.6 MB\t92750\tBWhair spatial dataset
S3-WB-48Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t208.6 MB\t58893\tBWhair spatial dataset
J4-W-19Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t288.8 MB\t60058\tBWhair spatial dataset
J1-W-41Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t228.2 MB\t29940\tBWhair spatial dataset
S14-W-70Y-M-3_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t111.7 MB\t19857\tBWhair spatial dataset
S14-B-70Y-M-2_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t245.9 MB\t45478\tBWhair spatial dataset
S13-B-69Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t452.2 MB\t81312\tBWhair spatial dataset
J5-W-23Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t230.3 MB\t52453\tBWhair spatial dataset
S10-W-63Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t656.0 MB\t133092\tBWhair spatial dataset
V2-W-58Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t350.1 MB\t64105\tBWhair spatial dataset
V6-WB-12Y-M_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t406.5 MB\t62244\tBWhair spatial dataset
J7-W-34Y-F_SCT.h5ad\tStereo-seq\t/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial\t227.1 MB\t48425\tBWhair spatial dataset
BWhair_scRNA.h5ad\tscRNA-seq\t/Files/ResultData/VirtualCell_BWhair/01.VirtualCell_BWhair_scRNA_scATAC\t10.8 GB\t764557\tBWhair scRNA dataset
BWhair_scATAC.h5ad\tscATAC-seq\t/Files/ResultData/VirtualCell_BWhair/01.VirtualCell_BWhair_scRNA_scATAC\t3.0 GB\t278792\tBWhair scATAC dataset
"""


HAIR_STATE_LABELS = {
    "B": "black",
    "W": "white",
    "WB": "black_white",
    "G": "gray",
    "GB": "gray_black",
}


def _read_embedded_tsv(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text.strip()), sep="\t")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(path.expanduser())
    return out


def _vcf_stem(path: str | Path) -> str:
    name = Path(str(path)).name
    for suffix in (".genotyper.vcf.gz", ".vcf.gz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _parse_size_mb(value: Any) -> float | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    parts = text.split()
    try:
        number = float(parts[0])
    except (ValueError, IndexError):
        return None
    unit = parts[1] if len(parts) > 1 else "MB"
    if unit.startswith("GB"):
        return number * 1024.0
    if unit.startswith("KB"):
        return number / 1024.0
    return number


def _normalise_sex(value: Any) -> str:
    token = str(value or "").strip().upper()
    if token.startswith("M"):
        return "M"
    if token.startswith("F"):
        return "F"
    return str(value or "").strip()


def _age_to_int(value: Any) -> int | None:
    extracted = pd.Series([str(value or "")]).str.extract(r"(\d+)", expand=False).iloc[0]
    if pd.isna(extracted):
        return None
    return int(extracted)


def _candidate_vcf_dirs() -> list[Path]:
    envs = [
        "VC_WGS_VCF_DIR",
        "VC_VIRTUAL_VCF_DIR",
        "VC_BW_WGS_VCF_DIR",
    ]
    paths = [Path(os.getenv(name, "")) for name in envs if os.getenv(name, "").strip()]
    paths.extend([
        Path("input/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("input/Files/ResultData/BW_WGS_vcf"),
        Path("/Files/ResultData/VirtualCell_WGS_vcf"),
        Path("/Files/ResultData/BW_WGS_vcf"),
    ])
    return _dedupe_paths(paths)


def _candidate_h5ad_dirs(modality: str) -> list[Path]:
    envs = ["VC_BWHAIR_H5AD_DIR", "VC_BWHAIR_ROOT", "VIRTUALCELL_BWHAIR_ROOT"]
    if modality == "spatial":
        envs.insert(0, "VC_BWHAIR_SPATIAL_DIR")
    if modality in {"scrna", "scatac"}:
        envs.insert(0, "VC_BWHAIR_SINGLECELL_DIR")
    paths = [Path(os.getenv(name, "")) for name in envs if os.getenv(name, "").strip()]
    root_expansions: list[Path] = []
    for path in paths:
        root_expansions.extend([
            path / "00.VirtualCell_BWhair_Spatial",
            path / "01.VirtualCell_BWhair_scRNA_scATAC",
        ])
    paths.extend(root_expansions)
    paths.extend([
        Path("input/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial"),
        Path("input/Files/ResultData/VirtualCell_BWhair/01.VirtualCell_BWhair_scRNA_scATAC"),
        Path("/Files/ResultData/VirtualCell_BWhair/00.VirtualCell_BWhair_Spatial"),
        Path("/Files/ResultData/VirtualCell_BWhair/01.VirtualCell_BWhair_scRNA_scATAC"),
    ])
    return _dedupe_paths(paths)


def _resolve_existing_file(filename: str, raw_path: str, candidate_dirs: list[Path]) -> str:
    raw = Path(str(raw_path or ""))
    candidates = []
    if filename:
        candidates.append(raw / filename)
    if not filename or raw.name == filename or raw.suffix:
        candidates.append(raw)
    for base in candidate_dirs:
        if filename:
            candidates.append(base / filename)
        candidates.append(base / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else str(raw)


def _infer_modality(technology: str, filename: str) -> str:
    text = f"{technology} {filename}".lower()
    if "atac" in text:
        return "scatac"
    if "scrna" in text or "rna" in text:
        return "scrna"
    if "stereo" in text or "spatial" in text:
        return "spatial"
    return "h5ad"


def _parse_bwhair_filename(filename: str) -> dict[str, Any]:
    sample_id = Path(filename).name.removesuffix(".h5ad").removesuffix("_SCT")
    parts = sample_id.split("-")
    if len(parts) < 4 or sample_id.startswith("BWhair_"):
        return {
            "sample_id": sample_id,
            "donor": "",
            "hair_state": "",
            "hair_state_label": "",
            "age": None,
            "sex": "",
            "replicate": "",
        }
    hair_state = parts[1]
    return {
        "sample_id": sample_id,
        "donor": parts[0],
        "hair_state": hair_state,
        "hair_state_label": HAIR_STATE_LABELS.get(hair_state, hair_state),
        "age": _age_to_int(parts[2]),
        "sex": _normalise_sex(parts[3]),
        "replicate": "-".join(parts[4:]),
    }


def load_default_wgs_manifest() -> pd.DataFrame:
    """Return the embedded VirtualCell WGS manifest with local path resolution."""
    df = _read_embedded_tsv(DEFAULT_WGS_MANIFEST_TSV)
    out = pd.DataFrame()
    out["sample_id"] = [_vcf_stem(v) for v in df["filePath"]]
    out["donor"] = df["Donor"].astype(str)
    out["part"] = df["Part"].astype(str)
    out["age"] = pd.Series([_age_to_int(v) for v in df["Age"]], dtype="Int64")
    out["sex"] = [_normalise_sex(v) for v in df["Sex"]]
    out["is_male"] = (out["sex"] == "M").astype(int)
    out["phenotype"] = df["Phenotype"].astype(str)
    first = out["donor"].astype(str).str[0].str.upper()
    phenotype_group = out["phenotype"].str.extract(r"^(Juvenile|Senile|Vitiligo)", expand=False)
    out["phenotype_group"] = phenotype_group.map({"Juvenile": "J", "Senile": "S", "Vitiligo": "V"}).fillna(first)
    out["vcf_path"] = [
        _resolve_existing_file(Path(v).name, str(v), _candidate_vcf_dirs())
        for v in df["filePath"]
    ]
    out["source_sample_id"] = df["sampleID"].astype(str)
    out["file_exists"] = [Path(p).exists() for p in out["vcf_path"]]
    out["has_index"] = [Path(str(p) + ".tbi").exists() for p in out["vcf_path"]]
    return out


def load_bwhair_manifest() -> pd.DataFrame:
    """Return the embedded BWhair h5ad manifest with parsed sample metadata."""
    raw = _read_embedded_tsv(DEFAULT_BWHAIR_MANIFEST_TSV)
    rows: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        filename = str(row["Filename"])
        modality = _infer_modality(str(row.get("Technology", "")), filename)
        parsed = _parse_bwhair_filename(filename)
        file_path = _resolve_existing_file(filename, str(row.get("Path", "")), _candidate_h5ad_dirs(modality))
        path = Path(file_path)
        stat_size_mb = round(path.stat().st_size / 1048576, 3) if path.exists() else None
        rows.append({
            "filename": filename,
            "sample_id": parsed["sample_id"],
            "donor": parsed["donor"],
            "hair_state": parsed["hair_state"],
            "hair_state_label": parsed["hair_state_label"],
            "age": parsed["age"],
            "sex": parsed["sex"],
            "replicate": parsed["replicate"],
            "technology": str(row.get("Technology", "")),
            "modality": modality,
            "path_dir": str(row.get("Path", "")),
            "file_path": file_path,
            "size": str(row.get("Size", "")),
            "size_mb": _parse_size_mb(row.get("Size")),
            "cells": int(row.get("Cells", 0) or 0),
            "description": str(row.get("Description", "")),
            "file_exists": path.exists(),
            "stat_size_mb": stat_size_mb,
        })
    return pd.DataFrame(rows)


def load_virtualcell_manifest(modalities: str = "all") -> pd.DataFrame:
    """Return a unified sample-level manifest across WGS and h5ad modalities."""
    requested = {m.strip().lower() for m in str(modalities or "all").split(",") if m.strip()}
    include_all = not requested or "all" in requested
    frames: list[pd.DataFrame] = []
    if include_all or "wgs" in requested:
        wgs = load_default_wgs_manifest().copy()
        wgs["modality"] = "wgs"
        wgs["technology"] = "WGS"
        wgs["file_path"] = wgs["vcf_path"]
        wgs["hair_state"] = ""
        wgs["hair_state_label"] = ""
        wgs["cells"] = pd.NA
        frames.append(wgs)
    h5ad = load_bwhair_manifest()
    if include_all:
        frames.append(h5ad)
    else:
        selected = h5ad[h5ad["modality"].isin(requested)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def virtualcell_inventory() -> dict[str, Any]:
    """Return compact inventory counts and file-readiness summaries."""
    wgs = load_default_wgs_manifest()
    h5ad = load_bwhair_manifest()
    h5ad_by_modality = h5ad.groupby("modality").agg(
        n_files=("filename", "count"),
        n_existing=("file_exists", "sum"),
        cells=("cells", "sum"),
        manifest_size_mb=("size_mb", "sum"),
    )
    return {
        "wgs": {
            "n_samples": int(len(wgs)),
            "n_existing": int(wgs["file_exists"].sum()),
            "n_indexed": int(wgs["has_index"].sum()),
            "phenotype_counts": wgs["phenotype"].value_counts().to_dict(),
            "donors": sorted(wgs["donor"].dropna().astype(str).unique().tolist()),
        },
        "h5ad": {
            "n_files": int(len(h5ad)),
            "n_existing": int(h5ad["file_exists"].sum()),
            "modalities": h5ad_by_modality.reset_index().to_dict(orient="records"),
            "donors": sorted([d for d in h5ad["donor"].dropna().astype(str).unique().tolist() if d]),
            "hair_state_counts": h5ad["hair_state"].value_counts().to_dict(),
        },
    }


def link_wgs_to_bwhair() -> pd.DataFrame:
    """Join WGS donors to BWhair h5ad coverage by donor."""
    wgs = load_default_wgs_manifest()
    h5ad = load_bwhair_manifest()
    sample_h5ad = h5ad[h5ad["donor"].astype(str) != ""].copy()
    rows: list[dict[str, Any]] = []
    for donor in sorted(set(wgs["donor"]).union(sample_h5ad["donor"])):
        w = wgs[wgs["donor"] == donor]
        h = sample_h5ad[sample_h5ad["donor"] == donor]
        rows.append({
            "donor": donor,
            "has_wgs": not w.empty,
            "wgs_sample_id": ",".join(w["sample_id"].astype(str).tolist()),
            "phenotype": ",".join(sorted(w["phenotype"].dropna().astype(str).unique().tolist())),
            "modalities": ",".join(sorted(h["modality"].dropna().astype(str).unique().tolist())),
            "hair_states": ",".join(sorted(h["hair_state"].dropna().astype(str).unique().tolist())),
            "n_h5ad_files": int(len(h)),
            "h5ad_cells": int(h["cells"].fillna(0).sum()) if not h.empty else 0,
        })
    return pd.DataFrame(rows)


def inspect_h5ad_metadata(
    file_path: str | Path,
    *,
    include_obs_summary: bool = True,
    max_categories: int = 8,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect h5ad metadata in backed mode, returning warnings instead of raising."""
    path = Path(file_path)
    cache_root = Path(cache_dir or os.getenv("VC_H5AD_METADATA_CACHE_DIR", "")).expanduser() if (cache_dir or os.getenv("VC_H5AD_METADATA_CACHE_DIR", "")) else None
    cache_path: Path | None = None
    if cache_root is not None and path.exists():
        try:
            import hashlib

            stat = path.stat()
            key_payload = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "obs": include_obs_summary,
                "max_categories": max_categories,
            }
            key = hashlib.sha256(repr(key_payload).encode("utf-8")).hexdigest()[:24]
            cache_path = cache_root / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached["cache_hit"] = True
                return cached
        except Exception:
            cache_path = None
    result: dict[str, Any] = {
        "file_path": str(path),
        "exists": path.exists(),
        "status": "missing" if not path.exists() else "ok",
        "cache_hit": False,
        "n_obs": None,
        "n_vars": None,
        "obs_columns": [],
        "var_columns": [],
        "obsm_keys": [],
        "varm_keys": [],
        "layers": [],
        "uns_keys": [],
        "obs_summary": {},
        "warnings": [],
    }
    if not path.exists():
        result["warnings"].append("h5ad file is not readable at the resolved path")
        return result
    if importlib.util.find_spec("anndata") is None:
        result["status"] = "dependency_missing"
        result["warnings"].append("anndata is not installed; install biobank-agent[single-cell] for h5ad inspection")
        return result

    adata = None
    try:
        import anndata as ad

        adata = ad.read_h5ad(path, backed="r")
        result["n_obs"] = int(adata.n_obs)
        result["n_vars"] = int(adata.n_vars)
        result["obs_columns"] = [str(c) for c in list(adata.obs.columns)]
        result["var_columns"] = [str(c) for c in list(adata.var.columns)]
        result["obsm_keys"] = [str(k) for k in list(getattr(adata, "obsm", {}).keys())[:50]]
        result["varm_keys"] = [str(k) for k in list(getattr(adata, "varm", {}).keys())[:50]]
        result["layers"] = [str(k) for k in list(getattr(adata, "layers", {}).keys())[:50]]
        result["uns_keys"] = [str(k) for k in list(adata.uns.keys())[:50]]
        if include_obs_summary:
            obs_summary: dict[str, Any] = {}
            for col in list(adata.obs.columns)[:30]:
                series = adata.obs[col]
                n_unique = int(series.nunique(dropna=True))
                item: dict[str, Any] = {"n_unique": n_unique, "n_missing": int(series.isna().sum())}
                if n_unique <= max_categories:
                    item["counts"] = {str(k): int(v) for k, v in series.value_counts(dropna=False).head(max_categories).items()}
                obs_summary[str(col)] = item
            result["obs_summary"] = obs_summary
    except Exception as exc:
        result["status"] = "error"
        result["warnings"].append(str(exc))
    finally:
        try:
            if adata is not None and hasattr(adata, "file"):
                adata.file.close()
        except Exception:
            pass
    if cache_path is not None and result.get("status") == "ok":
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return result


def h5ad_sample_summaries(
    *,
    modality: str = "all",
    sample_query: str = "",
    max_files: int = 5,
    include_obs_summary: bool = True,
) -> list[dict[str, Any]]:
    """Inspect selected h5ad samples and merge manifest metadata."""
    manifest = load_bwhair_manifest()
    modality_key = str(modality or "all").lower().strip()
    if modality_key and modality_key != "all":
        manifest = manifest[manifest["modality"] == modality_key]
    query = str(sample_query or "").lower().strip()
    if query:
        mask = (
            manifest["sample_id"].astype(str).str.lower().str.contains(query, regex=False)
            | manifest["filename"].astype(str).str.lower().str.contains(query, regex=False)
            | manifest["donor"].astype(str).str.lower().str.contains(query, regex=False)
        )
        manifest = manifest[mask]
    if max_files > 0:
        manifest = manifest.head(max_files)
    summaries = []
    for row in manifest.to_dict(orient="records"):
        meta = inspect_h5ad_metadata(row["file_path"], include_obs_summary=include_obs_summary)
        summaries.append({
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "donor": row["donor"],
            "hair_state": row["hair_state"],
            "modality": row["modality"],
            "technology": row["technology"],
            "manifest_cells": row["cells"],
            "manifest_size_mb": row["size_mb"],
            **meta,
        })
    return summaries


__all__ = [
    "DEFAULT_WGS_MANIFEST_TSV",
    "DEFAULT_BWHAIR_MANIFEST_TSV",
    "HAIR_STATE_LABELS",
    "h5ad_sample_summaries",
    "inspect_h5ad_metadata",
    "link_wgs_to_bwhair",
    "load_bwhair_manifest",
    "load_default_wgs_manifest",
    "load_virtualcell_manifest",
    "virtualcell_inventory",
]
