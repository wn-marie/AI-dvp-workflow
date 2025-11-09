import argparse
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "Unknown",
    7: "Trauma Center",
    8: "Not Mapped",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Home",
    2: "Another hospital",
    3: "Skilled nursing facility",
    4: "Intermediate care facility",
    5: "Other inpatient care",
    6: "Home health service",
    7: "Left AMA",
    8: "Home IV care",
    9: "Admitted to hospital",
    10: "Neonate aftercare",
    11: "Expired",
    12: "Still patient",
    13: "Hospice home",
    14: "Hospice medical",
    15: "Medicare swing bed",
    16: "Outpatient services other institution",
    17: "Outpatient services this institution",
    18: "Unknown",
    19: "Expired home hospice",
    20: "Expired medical hospice",
    21: "Expired unknown place",
    22: "Rehab",
    23: "Long-term care hospital",
    24: "Medicaid nursing facility",
    25: "Not mapped",
    26: "Unknown/Invalid",
    27: "Federal facility",
    28: "Psychiatric hospital",
    29: "Critical access hospital",
    30: "Other healthcare institution",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician referral",
    2: "Clinic referral",
    3: "HMO referral",
    4: "Transfer hospital",
    5: "Transfer SNF",
    6: "Transfer other facility",
    7: "Emergency room",
    8: "Court/law enforcement",
    9: "Not available",
    10: "Transfer critical access hospital",
    11: "Normal delivery",
    12: "Premature delivery",
    13: "Sick baby",
    14: "Extramural birth",
    15: "Not available ",
    17: "NULL",
    18: "Transfer other facility",
    19: "Unknown",
    20: "Not mapped",
    21: "Birthing center",
    22: "Transfer psychiatric hospital",
    23: "Transfer rehabilitation",
    24: "Not known",
    25: "Transfer another health care facility",
    26: "Clinic referral - other",
}

CHRONIC_CONDITION_RANGES = {
    "circulatory": [(390, 459), (785, 785)],
    "respiratory": [(460, 519), (786, 786)],
    "digestive": [(520, 579)],
    "diabetes": [(250, 250)],
    "injury": [(800, 999)],
    "musculoskeletal": [(710, 739)],
    "genitourinary": [(580, 629)],
    "neoplasms": [(140, 239)],
    "mental": [(290, 319)],
    "neurological": [(320, 389)],
    "infectious": [(1, 139)],
}

EXPIRE_DISPOSITION_IDS = {11, 19, 20, 21}
HIGH_RISK_DISCHARGE_IDS = {
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    14,
    18,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    30,
}
EMERGENCY_ADMISSION_IDS = {1, 2, 7}
EMERGENCY_SOURCE_IDS = {7}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

MED_STATUS_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": -1}


def load_raw_datasets(paths: Iterable[Path]) -> pd.DataFrame:
    """Load and combine one or more raw CSV files."""
    dataframes: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            print(f"Warning: input file {path} not found. Skipping.")
            continue
        df = pd.read_csv(path)
        df["_source_file"] = path.name
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            "No input CSV files could be loaded. "
            "Verify the provided --input-path arguments."
        )

    combined = pd.concat(dataframes, ignore_index=True, sort=False)
    if "encounter_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["encounter_id"], keep="last")
    return combined


def diagnosis_to_categories(diag_values: Iterable[str]) -> Set[str]:
    categories: Set[str] = set()
    for value in diag_values:
        if pd.isna(value):
            continue
        # Some diagnosis codes start with V or E – treat those as separate categories
        if isinstance(value, str) and value.startswith(("V", "E")):
            categories.add(value[0])
            continue
        try:
            code = float(value)
        except (TypeError, ValueError):
            continue
        code_int = int(code)
        for category, ranges in CHRONIC_CONDITION_RANGES.items():
            for low, high in ranges:
                if low <= code_int <= high:
                    categories.add(category)
                    break
    return categories


def diagnosis_primary_category(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, str) and value.startswith(("V", "E")):
        return value[0]
    try:
        code = float(value)
    except (TypeError, ValueError):
        return "Other"
    code_int = int(code)
    for category, ranges in CHRONIC_CONDITION_RANGES.items():
        for low, high in ranges:
            if low <= code_int <= high:
                return category
    if 240 <= code_int <= 279:
        return "endocrine"
    if 280 <= code_int <= 289:
        return "blood"
    if 290 <= code_int <= 319:
        return "mental"
    if 320 <= code_int <= 389:
        return "neurological"
    if 520 <= code_int <= 579:
        return "digestive"
    if 580 <= code_int <= 629:
        return "genitourinary"
    if 630 <= code_int <= 679:
        return "pregnancy"
    if 680 <= code_int <= 709:
        return "skin"
    if 710 <= code_int <= 739:
        return "musculoskeletal"
    if 740 <= code_int <= 759:
        return "congenital"
    if 780 <= code_int <= 799:
        return "symptoms"
    if 800 <= code_int <= 999:
        return "injury"
    return "Other"


def convert_age_to_numeric(age_range: str) -> float:
    if pd.isna(age_range):
        return pd.NA
    parts = str(age_range).replace("[", "").replace(")", "").split("-")
    if len(parts) != 2:
        return pd.NA
    try:
        low, high = map(int, parts)
    except ValueError:
        return pd.NA
    return (low + high) / 2


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["previous_admissions"] = (
        df["number_inpatient"].fillna(0)
        + df["number_outpatient"].fillna(0)
        + df["number_emergency"].fillna(0)
    )

    columns_to_map = {
        "admission_type_id": ADMISSION_TYPE_MAP,
        "discharge_disposition_id": DISCHARGE_DISPOSITION_MAP,
        "admission_source_id": ADMISSION_SOURCE_MAP,
    }
    for column, mapping in columns_to_map.items():
        df[f"{column}_desc"] = df[column].map(mapping).fillna("Other")

    df["age_num"] = df["age"].apply(convert_age_to_numeric)

    comorbidity_categories = df[["diag_1", "diag_2", "diag_3"]].apply(
        lambda row: diagnosis_to_categories(row.values), axis=1
    )
    df["comorbidity_index"] = comorbidity_categories.apply(len)

    med_status = df[MEDICATION_COLUMNS].fillna("No")
    df["medications_active_count"] = (med_status != "No").sum(axis=1)
    df["medications_change_count"] = med_status.isin(["Up", "Down"]).sum(axis=1)
    df["medications_up_count"] = med_status.eq("Up").sum(axis=1)
    df["medications_down_count"] = med_status.eq("Down").sum(axis=1)
    df["medications_intensity_score"] = med_status.replace(MED_STATUS_MAP).sum(axis=1)
    df["insulin_adjusted_flag"] = med_status["insulin"].isin(["Up", "Down"]).astype(int)
    df["change_flag"] = (df["change"] == "Ch").astype(int)
    df["diabetes_med_flag"] = (df["diabetesMed"] == "Yes").astype(int)

    df["diag_1_category"] = df["diag_1"].apply(diagnosis_primary_category)
    df["diag_2_category"] = df["diag_2"].apply(diagnosis_primary_category)
    df["diag_3_category"] = df["diag_3"].apply(diagnosis_primary_category)

    df["high_risk_discharge_flag"] = df["discharge_disposition_id"].isin(
        HIGH_RISK_DISCHARGE_IDS
    ).astype(int)
    df["emergency_admission_flag"] = df["admission_type_id"].isin(
        EMERGENCY_ADMISSION_IDS
    ).astype(int)
    df["er_source_flag"] = df["admission_source_id"].isin(EMERGENCY_SOURCE_IDS).astype(int)

    df["previous_inpatient_flag"] = (df["number_inpatient"] > 0).astype(int)
    df["previous_emergency_flag"] = (df["number_emergency"] > 0).astype(int)
    df["previous_outpatient_flag"] = (df["number_outpatient"] > 0).astype(int)
    df["lab_to_med_ratio"] = df["num_lab_procedures"] / df["num_medications"].replace(
        0, pd.NA
    )
    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"].replace(
        0, pd.NA
    )
    df["medications_per_day"] = df["num_medications"] / df["time_in_hospital"].replace(
        0, pd.NA
    )

    return df


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.replace("?", pd.NA)

    # Remove rows where patient died or was discharged to hospice since readmission is not applicable
    df = df[~df["discharge_disposition_id"].isin(EXPIRE_DISPOSITION_IDS)].copy()

    # Drop encounters missing the target label
    df = df[df["readmitted"].notna()].copy()

    df["readmitted_30_days"] = (df["readmitted"] == "<30").astype(int)
    df = engineer_features(df)

    df["medical_specialty_grouped"] = (
        df["medical_specialty"]
        .fillna("Unknown")
        .apply(lambda value: value.strip() if isinstance(value, str) else value)
    )
    top_specialties = (
        df["medical_specialty_grouped"]
        .value_counts(dropna=True)
        .nlargest(15)
        .index.tolist()
    )
    df["medical_specialty_grouped"] = df["medical_specialty_grouped"].apply(
        lambda value: value if value in top_specialties else "Other"
    )

    # Drop identifiers unlikely to generalize
    drop_columns = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "readmitted",
        "_source_file",
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    df["lab_to_med_ratio"] = df["lab_to_med_ratio"].fillna(0)
    df["procedures_per_day"] = df["procedures_per_day"].fillna(0)
    df["medications_per_day"] = df["medications_per_day"].fillna(0)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and feature engineer the diabetic readmission dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        action="append",
        dest="input_paths",
        help=(
            "Path to a raw CSV file. Provide multiple times to combine datasets. "
            "Defaults to the bundled diabetic_data and diabetic_data2 files when omitted."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/diabetic_data_processed.csv"),
        help="Path to save the processed CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    default_input_paths = [
        Path("data/raw/diabetic_data.csv"),
        Path("data/raw/diabetic_data2.csv"),
    ]
    input_paths = args.input_paths if args.input_paths else default_input_paths

    df = load_raw_datasets(input_paths)
    processed_df = clean_data(df)
    processed_df.to_csv(args.output_path, index=False)
    print(f"Saved processed data to {args.output_path} with {processed_df.shape[0]} rows.")


if __name__ == "__main__":
    main()

