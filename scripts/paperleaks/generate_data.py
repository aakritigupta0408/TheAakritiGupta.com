#!/usr/bin/env python3
"""Regenerate public/paperleaks/data.js from the canonical CSV.

The CSV (public/paperleaks/india_paper_leaks_2014_2026.csv) is the source of
truth. This script derives the Category / Outcome / Education Level columns,
writes them back into the CSV, and emits every aggregate the site reads.

Usage: python3 scripts/paperleaks/generate_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "public/paperleaks/india_paper_leaks_2014_2026.csv"
OUT = ROOT / "public/paperleaks/data.js"

UPDATED = "22 July 2026"  # bump when the dataset is refreshed

SRC_KAGGLE = "Kaggle original dataset"


def category(row) -> str:
    t = (str(row["Exam Name"]) + " " + str(row["Conducting Body"])).lower()
    if re.search(r"neet|aipmt|cpmt|pre-medical|fmge|mbbs|medical|nbems|health officer|\bcho\b|pharmacy|nursing", t):
        return "Medical"
    if re.search(r"jee|engineering|\bvit\b|junior engineer|\bje\b|polytechnic", t):
        return "Engineering"
    if re.search(r"\btet\b|teacher|b\.ed|elementary education|d\.el\.ed|assistant professor|ugc net|\bnet\b|otet|education service", t):
        return "Teacher & Academia"
    if re.search(r"police|constable|\bsi\b|sub.inspector|excise|warder|jail|army|agniveer", t):
        return "Police & Defence"
    if re.search(r"civil services|upsc|bpsc|\bpsc\b|public service|\bcgl\b|combined competitive|review officer|\baro\b|clerk|patwari|\bamin\b|\bvdo\b|aedo", t):
        return "Civil Services & Govt Jobs"
    if re.search(r"class 9|class 10|class 11|class 12|matric|hslc|hsslc|sslc|madhyamik|intermediate|board exam|hsc|puc|high school|higher secondary|10th|12th|pre-board|cbse|icse|hs |board", t) and "recruitment" not in t:
        return "School Board"
    if re.search(r"civil services|upsc|bpsc|psc |public service|cgl|combined competitive|review officer|aro|clerk|patwari|amin|vdo|aedo|revenue", t):
        return "Civil Services & Govt Jobs"
    if re.search(r"university|mba|bcom|b\.com|b\.tech|btech|semester|\bsem\b|degree|llb|bba|bcs", t):
        return "University"
    return "Other Recruitment"


def outcome(row) -> str:
    t = (str(row["Action taken"]) + " " + str(row["Note about action Taken"]) + " " + str(row["Note about incident"])).lower()
    if re.search(r"re-?exam(?!.*no)|re-?test|re-?conduct|held again|re-?held|alternative question paper", t) and not re.search(r"no re-?exam|no rexam", t):
        return "Re-exam held or ordered"
    if re.search(r"cancell?ed|scrapped|postponed", t):
        return "Cancelled/Postponed, no re-exam recorded"
    if re.search(r"dismissed|denied|no evidence|fake|baseless|rumour|refuted|no rexam|no re-?exam", t):
        return "No judgement — dismissed or no action"
    return "Legal/administrative action only"


LEVELS = [
    "School — Class 10",
    "School — Class 12",
    "School — other/combined",
    "Undergraduate entrance & courses",
    "Postgraduate & professional",
    "Job & recruitment exams",
]

# Explicit per-exam overrides where name rules are ambiguous. Keyed on a
# substring of the exam name (lowercased).
LEVEL_OVERRIDES = {
    "b.ed. 4th sem": "Postgraduate & professional",
    "mba": "Postgraduate & professional",
    "foreign medical graduate": "Postgraduate & professional",
    "community health officer": "Job & recruitment exams",
    "nursing officer": "Job & recruitment exams",
    "diploma in elementary education": "Postgraduate & professional",
    "high school english exam": "School — Class 10",
    # Odisha BSE Class 10 Modern Indian Language paper
    "mother india language": "School — Class 10",
    # MSBSHSE SSC Class 10 combined History & Political Science paper
    "history and political science": "School — Class 10",
}


def level(row) -> str:
    name = str(row["Exam Name"]).lower()
    cat = row["Category"]
    for key, lv in LEVEL_OVERRIDES.items():
        if key in name:
            return lv
    if cat in ("Civil Services & Govt Jobs", "Police & Defence", "Other Recruitment"):
        return "Job & recruitment exams"
    if cat == "Teacher & Academia":
        # NET/SET/JRF are post-PG academic eligibility; TETs & recruitment are jobs
        if re.search(r"ugc net|csir|net 20|\bset\b|assistant professor", name):
            return "Postgraduate & professional"
        return "Job & recruitment exams"
    if cat == "Medical":
        if re.search(r"neet[- ]?pg|\bpg\b|fmge|\bmd\b|\bms\b", name):
            return "Postgraduate & professional"
        if re.search(r"neet|aipmt|cpmt|ug", name):
            return "Undergraduate entrance & courses"
        return "Undergraduate entrance & courses"
    if cat == "Engineering":
        if re.search(r"junior engineer|\bje\b", name):
            return "Job & recruitment exams"
        return "Undergraduate entrance & courses"
    if cat == "University":
        if re.search(r"mba|llm|m\.|pg |post.?grad", name):
            return "Postgraduate & professional"
        return "Undergraduate entrance & courses"
    if cat == "School Board":
        has10 = bool(re.search(r"class 10\b|10th", name))
        has12 = bool(re.search(r"class 12\b|12th", name))
        other = bool(re.search(r"class 9\b|class 11\b|first year", name))
        if (has10 and has12) or (other and (has10 or has12)):
            return "School — other/combined"
        if other:
            return "School — other/combined"
        if has10:
            return "School — Class 10"
        if has12:
            return "School — Class 12"
        if re.search(r"matric|hslc\b|\bsslc\b|madhyamik|high school|secondary school", name):
            return "School — Class 10"
        if re.search(r"intermediate|hsslc|\bpuc\b|plus one|plus two|higher secondary|\bhsc\b|\binter\b", name):
            return "School — Class 12"
        return "School — other/combined"
    return "School — other/combined"


def main() -> None:
    df = pd.read_csv(CSV)
    df["year"] = pd.to_datetime(df["Date of Exam/Incident"], format="%d-%m-%Y").dt.year
    # Researched candidate counts (added Jul 2026) take precedence over the
    # sparse original Appeared Students column.
    if "Students Affected (Researched)" in df.columns:
        df["Students"] = df["Students Affected (Researched)"].fillna(df["Appeared Students"])
    else:
        df["Students"] = df["Appeared Students"]
    df["Category"] = df.apply(category, axis=1)
    df["Outcome"] = df.apply(outcome, axis=1)
    df["Education Level"] = df.apply(level, axis=1)

    unmatched = df[~df["Education Level"].isin(LEVELS)]
    if len(unmatched):
        raise SystemExit(f"Unclassified levels:\n{unmatched[['Exam Name']].to_string()}")

    df.to_csv(CSV, index=False)

    years = list(range(2014, 2027))
    st = df.dropna(subset=["Students"])
    cats = df["Category"].value_counts().index.tolist()
    levels_present = [l for l in LEVELS if (df["Education Level"] == l).any()]
    confirmed = df["Leak Confirmation Status"].str.strip() == "confirmed"

    def by_year(mask) -> list:
        return [int(((df.year == y) & mask).sum()) for y in years]

    data = {
        "meta": {
            "total": len(df),
            "confirmed": int(confirmed.sum()),
            "studentsTotal": int(st["Students"].sum()),
            "studentsRows": len(st),
            "updated": UPDATED,
        },
        "years": years,
        "confirmedPerYear": by_year(confirmed),
        "accusedPerYear": by_year(~confirmed),
        "perState": df["Area(s) of Incident"].str.strip().value_counts().to_dict(),
        "perCategory": df["Category"].value_counts().to_dict(),
        "categoryYearMatrix": {c: by_year(df.Category == c) for c in cats},
        "perLevel": {l: int((df["Education Level"] == l).sum()) for l in levels_present},
        "levelYearMatrix": {l: by_year(df["Education Level"] == l) for l in levels_present},
        "outcomes": df["Outcome"].value_counts().to_dict(),
        "outcomePerYear": {o: by_year(df.Outcome == o) for o in df["Outcome"].unique()},
        "studentsByYear": {int(y): int(v) for y, v in st.groupby("year")["Students"].sum().items()},
        "studentsByCategory": {c: int(v) for c, v in st.groupby("Category")["Students"].sum().sort_values(ascending=False).items()},
        "studentsByCategoryYear": {c: {int(y): int(v) for y, v in g.groupby("year")["Students"].sum().items()} for c, g in st.groupby("Category")},
        "incidents": df[[
            "Date of Exam/Incident", "Exam Name", "Area(s) of Incident", "Category",
            "Education Level", "Leak Confirmation Status", "Outcome", "Students",
            "References", "Record Source",
        ]].fillna("").to_dict("records"),
    }
    OUT.write_text("window.PAPERLEAK_DATA = " + json.dumps(data) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("levels:", data["perLevel"])


if __name__ == "__main__":
    main()
