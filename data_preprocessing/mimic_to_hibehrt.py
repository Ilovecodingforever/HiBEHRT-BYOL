import argparse
import json
import pickle
import re
from pathlib import Path

import pandas as pd


SPECIAL_TOKENS = ["PAD", "UNK", "MASK", "SEP"]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    repos_root = repo_root.parent

    parser = argparse.ArgumentParser(
        description="Convert MIMIC-IV pipeline outputs to HiBEHRT-BYOL patient sequence files."
    )
    parser.add_argument(
        "--mimic-data-dir",
        default=str(repos_root / "MIMIC-IV-Data-Pipeline" / "data"),
        help="Path to the MIMIC-IV-Data-Pipeline data directory.",
    )
    parser.add_argument(
        "--patients",
        default=str(
            repos_root
            / "MIMIC-IV-Data-Pipeline"
            / "mimiciv"
            / "3.1"
            / "hosp"
            / "patients.csv.gz"
        ),
        help="Raw MIMIC patients table for visit-level age calculation.",
    )
    parser.add_argument(
        "--cohort",
        default="cohort/cohort_non-icu_mortality_0_.csv",
        help="Cohort CSV/CSV.GZ path, absolute or relative to --mimic-data-dir.",
    )
    parser.add_argument(
        "--diag",
        default="features/preproc_diag.csv",
        help="Diagnosis feature CSV/CSV.GZ path, absolute or relative to --mimic-data-dir.",
    )
    parser.add_argument(
        "--med",
        default="features/preproc_med.csv",
        help="Medication feature CSV/CSV.GZ path, absolute or relative to --mimic-data-dir.",
    )
    parser.add_argument(
        "--proc",
        default="features/preproc_proc.csv",
        help="Procedure feature CSV/CSV.GZ path, absolute or relative to --mimic-data-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "data" / "mimic"),
        help="Directory for generated HiBEHRT-ready Parquet and dictionary files.",
    )
    parser.add_argument("--ssl-name", default="ssl.parquet")
    parser.add_argument("--labeled-name", default="labeled.parquet")
    parser.add_argument("--token-dict-name", default="dict4all")
    parser.add_argument("--age-dict-name", default="dict4age")
    return parser.parse_args()


def resolve_path(base_dir, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_token(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "missing"


def code_token(prefix, value):
    return f"{prefix}_{normalize_token(value).upper()}"


def med_token(value):
    return f"MED_{normalize_token(value)}"


def valid_dod(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"0", "nan", "nat", "none", "null"}


def make_feature_map(frame, value_col, token_fn, time_cols=None):
    if frame.empty:
        return {}

    time_cols = time_cols or []
    cols = ["hadm_id", value_col] + [col for col in time_cols if col in frame.columns]
    data = frame[cols].dropna(subset=["hadm_id", value_col]).copy()
    data["hadm_id"] = data["hadm_id"].astype(str)

    sort_cols = [col for col in time_cols if col in data.columns] + [value_col]
    if sort_cols:
        data = data.sort_values(sort_cols, kind="mergesort")

    data["token"] = data[value_col].map(token_fn)
    data = data.drop_duplicates(["hadm_id", "token"])
    return data.groupby("hadm_id")["token"].apply(list).to_dict()


def build_vocab(values):
    idx2token = {idx: token for idx, token in enumerate(SPECIAL_TOKENS)}
    next_idx = len(idx2token)
    for value in sorted(set(values)):
        if value in SPECIAL_TOKENS:
            continue
        idx2token[next_idx] = value
        next_idx += 1
    token2idx = {token: idx for idx, token in idx2token.items()}
    return {"token2idx": token2idx, "idx2token": idx2token}


def save_pickle(obj, output_dir, name):
    path = output_dir / f"{name}.pkl"
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def add_visit_age(cohort, patients):
    data = cohort.copy()
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    if patients is None:
        data["visit_age"] = data["Age"]
        return data

    required = {"subject_id", "anchor_age", "anchor_year"}
    missing = sorted(required - set(patients.columns))
    if missing:
        data["visit_age"] = data["Age"]
        return data

    patient_age = patients[list(required)].copy()
    patient_age["subject_id"] = patient_age["subject_id"].astype(str)
    patient_age["anchor_age"] = pd.to_numeric(patient_age["anchor_age"], errors="coerce")
    patient_age["anchor_year"] = pd.to_numeric(patient_age["anchor_year"], errors="coerce")
    patient_age["birth_year"] = patient_age["anchor_year"] - patient_age["anchor_age"]
    data = data.merge(patient_age[["subject_id", "birth_year"]], on="subject_id", how="left")
    calculated = data["admittime"].dt.year - data["birth_year"]
    data["visit_age"] = calculated.where(calculated.notna(), data["Age"])
    return data.drop(columns=["birth_year"])


def build_sequences(cohort, patients, diag_map, med_map, proc_map):
    required = {"subject_id", "hadm_id", "admittime", "Age", "dod"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Cohort is missing required columns: {missing}")

    data = cohort.copy()
    data["subject_id"] = data["subject_id"].astype(str)
    data["hadm_id"] = data["hadm_id"].astype(str)
    data["admittime"] = pd.to_datetime(data["admittime"])
    data = add_visit_age(data, patients)
    data["visit_age"] = pd.to_numeric(data["visit_age"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["subject_id", "hadm_id", "admittime", "visit_age"])
    data = data.sort_values(["subject_id", "admittime", "hadm_id"], kind="mergesort")

    rows = []
    for subject_id, group in data.groupby("subject_id", sort=False):
        codes = []
        ages = []

        for visit in group.itertuples(index=False):
            hadm_id = str(visit.hadm_id)
            visit_age = str(int(visit.visit_age))
            visit_tokens = []
            visit_tokens.extend(diag_map.get(hadm_id, []))
            visit_tokens.extend(med_map.get(hadm_id, []))
            visit_tokens.extend(proc_map.get(hadm_id, []))
            visit_tokens.append("SEP")

            codes.extend(visit_tokens)
            ages.extend([visit_age] * len(visit_tokens))

        label = int(group["dod"].map(valid_dod).any())
        rows.append(
            {
                "subject_id": subject_id,
                "code": codes,
                "age": ages,
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def validate_outputs(labeled, token_vocab, age_vocab):
    if labeled.empty:
        raise ValueError("No patient sequences were generated.")

    bad_lengths = labeled[labeled["code"].map(len) != labeled["age"].map(len)]
    if not bad_lengths.empty:
        raise ValueError(f"{len(bad_lengths)} rows have mismatched code/age lengths.")

    no_sep = labeled[~labeled["code"].map(lambda values: "SEP" in values)]
    if not no_sep.empty:
        raise ValueError(f"{len(no_sep)} rows have no SEP token.")

    labels = set(labeled["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0/1, found {sorted(labels)}.")

    token2idx = token_vocab["token2idx"]
    age2idx = age_vocab["token2idx"]
    missing_tokens = sorted(
        {token for seq in labeled["code"] for token in seq if token not in token2idx}
    )
    missing_ages = sorted({age for seq in labeled["age"] for age in seq if age not in age2idx})
    if missing_tokens:
        raise ValueError(f"Tokens missing from dictionary: {missing_tokens[:10]}")
    if missing_ages:
        raise ValueError(f"Ages missing from dictionary: {missing_ages[:10]}")


def main():
    args = parse_args()
    mimic_data_dir = Path(args.mimic_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = read_csv(resolve_path(mimic_data_dir, args.cohort))
    patients = read_csv(Path(args.patients)) if args.patients else None
    diag = read_csv(resolve_path(mimic_data_dir, args.diag))
    med = read_csv(resolve_path(mimic_data_dir, args.med))
    proc = read_csv(resolve_path(mimic_data_dir, args.proc))

    diag_map = make_feature_map(diag, "new_icd_code", lambda value: code_token("DIA", value))
    med_map = make_feature_map(
        med,
        "drug_name",
        med_token,
        time_cols=["starttime", "start_hours_from_admit"],
    )
    proc_map = make_feature_map(
        proc,
        "icd_code",
        lambda value: code_token("PRO", value),
        time_cols=["chartdate", "proc_time_from_admit"],
    )

    labeled = build_sequences(cohort, patients, diag_map, med_map, proc_map)
    ssl = labeled[["subject_id", "code", "age"]].copy()

    token_vocab = build_vocab(token for seq in labeled["code"] for token in seq)
    age_vocab = build_vocab(age for seq in labeled["age"] for age in seq)
    validate_outputs(labeled, token_vocab, age_vocab)

    ssl_path = output_dir / args.ssl_name
    labeled_path = output_dir / args.labeled_name
    ssl.to_parquet(ssl_path, index=False)
    labeled.to_parquet(labeled_path, index=False)
    token_dict_path = save_pickle(token_vocab, output_dir, args.token_dict_name)
    age_dict_path = save_pickle(age_vocab, output_dir, args.age_dict_name)

    summary = {
        "patients": int(len(labeled)),
        "positive_patients": int(labeled["label"].sum()),
        "negative_patients": int(len(labeled) - labeled["label"].sum()),
        "token_vocab_size": int(len(token_vocab["token2idx"])),
        "age_vocab_size": int(len(age_vocab["token2idx"])),
        "min_sequence_length": int(labeled["code"].map(len).min()),
        "max_sequence_length": int(labeled["code"].map(len).max()),
        "patients_with_age_changes": int(labeled["age"].map(lambda seq: len(set(seq)) > 1).sum()),
        "ssl_path": str(ssl_path),
        "labeled_path": str(labeled_path),
        "token_dict_path": str(token_dict_path),
        "age_dict_path": str(age_dict_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
