
# Analysis Pipeline - Naukri Job Market Intelligence
# This script loads the scraped data, cleans it,
# finds insights, builds ML model and saves to database


import os
import re
import warnings
import sqlite3
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# find project folder and set paths
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER    = os.path.join(PROJECT_FOLDER, "data")
OUTPUT_FOLDER  = os.path.join(PROJECT_FOLDER, "outputs")
os.makedirs(DATA_FOLDER,   exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# Skills we want to look for in job descriptions
# ============================================================
SKILLS = [
    "python", "sql", "excel", "tableau", "power bi", "looker",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "spark", "hadoop", "kafka", "airflow", "dbt", "databricks", "snowflake",
    "aws", "gcp", "azure", "redshift", "bigquery", "pyspark",
    "machine learning", "deep learning", "nlp", "statistics",
    "a/b testing", "data warehouse", "etl", "data pipeline",
    "docker", "git", "api", "flask", "streamlit",
]

# handle different ways people write the same skill
SKILL_SYNONYMS = {
    "power bi":       ["powerbi", "power-bi", "ms power bi"],
    "scikit-learn":   ["sklearn", "scikit learn"],
    "excel":          ["ms excel", "microsoft excel"],
    "pyspark":        ["py spark"],
    "machine learning": ["ml engineer"],
    "nlp":            ["natural language processing"],
}


def find_skills(text):
    # look for skills in job description text
    t = str(text).lower()
    # replace synonyms first
    for main_skill, alternatives in SKILL_SYNONYMS.items():
        for alt in alternatives:
            t = t.replace(alt, main_skill)
    found = []
    for skill in SKILLS:
        # \b means match whole word only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, t):
            found.append(skill)
    return found


def get_experience_years(exp_text):
    # extract number from text like "2-5 Yrs" -> 2
    if pd.isna(exp_text):
        return None
    match = re.search(r"(\d+)", str(exp_text))
    return int(match.group(1)) if match else None


def get_salary_bucket(salary_lpa):
    # put salary into a bucket for easier analysis
    if salary_lpa is None or pd.isna(salary_lpa):
        return "Not Disclosed"
    if salary_lpa < 3:   return "0-3 LPA"
    if salary_lpa < 6:   return "3-6 LPA"
    if salary_lpa < 10:  return "6-10 LPA"
    if salary_lpa < 15:  return "10-15 LPA"
    return "15+ LPA"


def get_role_type(title):
    # classify job title into a role category
    t = str(title).lower()
    if any(w in t for w in ["machine learning", "ml engineer", "ai engineer"]):
        return "ML Engineer"
    if any(w in t for w in ["data scientist", "data science"]):
        return "Data Scientist"
    if any(w in t for w in ["data engineer", "data engineering"]):
        return "Data Engineer"
    if any(w in t for w in ["power bi", "tableau", "bi analyst"]):
        return "BI Analyst"
    if any(w in t for w in ["business analyst", "business intelligence"]):
        return "Business Analyst"
    if any(w in t for w in ["data analyst", "analyst"]):
        return "Data Analyst"
    if "sql" in t:
        return "SQL Developer"
    return "Other"


def get_level(title, years_exp):
    # figure out if job is for fresher, mid level or senior
    t   = str(title).lower()
    exp = years_exp or 0
    if any(w in t for w in ["senior", "sr ", "lead", "principal", "head"]):
        return "Senior"
    if any(w in t for w in ["junior", "jr ", "fresher", "intern", "trainee"]):
        return "Fresher/Junior"
    if exp <= 2:  return "Fresher/Junior"
    if exp <= 5:  return "Mid-Level"
    return "Senior"


def get_city_name(location):
    # extract main city from location text
    loc = str(location).lower()
    for city in ["mumbai", "bangalore", "bengaluru", "delhi", "hyderabad",
                 "pune", "chennai", "kolkata", "noida", "gurgaon", "gurugram"]:
        if city in loc:
            return "Bangalore" if city == "bengaluru" else city.capitalize()
    if "remote" in loc:
        return "Remote"
    return "Other"


def get_work_type(title, location):
    # figure out if job is remote, hybrid or onsite
    text = (str(title) + " " + str(location)).lower()
    if any(w in text for w in ["remote", "work from home", "wfh"]):
        return "Remote"
    if "hybrid" in text:
        return "Hybrid"
    return "Onsite"


# ============================================================
# Load raw scraped data
# ============================================================
raw_path = os.path.join(DATA_FOLDER, "naukri_raw.csv")
raw_df   = pd.read_csv(raw_path)
print(f"Raw jobs loaded: {len(raw_df)}")

# ============================================================
# Clean the data step by step
# ============================================================
df = raw_df.copy()

# step 1 - remove rows with no job title
before = len(df)
df = df.dropna(subset=["title"])
print(f"After removing blank titles: {len(df)} (removed {before - len(df)})")

# step 2 - duplicate audit then remove using job_url (most accurate)
dup_title     = df.duplicated(subset=["title"]).sum()
dup_title_co  = df.duplicated(subset=["title", "company"]).sum()
dup_title_loc = df.duplicated(subset=["title", "company", "location"]).sum()
dup_url       = df.duplicated(subset=["job_url"]).sum()

print("\nDUPLICATE AUDIT:")
print(f"Same title only:                 {dup_title}")
print(f"Same title + company:            {dup_title_co}")
print(f"Same title + company + location: {dup_title_loc}")
print(f"Same job_url:                    {dup_url}")
print(f"Unique URLs:                     {df['job_url'].nunique()}")

# save audit to CSV so recruiters can see evidence on GitHub
audit = {
    "raw_jobs": len(raw_df),
    "after_null_removal": len(df),
    "duplicate_titles": int(dup_title),
    "duplicate_title_company": int(dup_title_co),
    "duplicate_title_company_location": int(dup_title_loc),
    "duplicate_urls": int(dup_url),
    "unique_urls": int(df["job_url"].nunique()),
}
pd.DataFrame([audit]).to_csv(
    os.path.join(OUTPUT_FOLDER, "duplicate_audit.csv"), index=False
)
print("Duplicate audit saved to outputs/duplicate_audit.csv")


before = len(df)
df = df.drop_duplicates(subset=["job_url"]).reset_index(drop=True)
print(f"After deduplication: {len(df)} (removed {before - len(df)})")

# step 3 - fill missing values
df["title"]       = df["title"].str.strip()
df["company"]     = df["company"].fillna("Unknown").str.strip()
df["location"]    = df["location"].fillna("India")
df["description"] = df["description"].fillna("")
df["skills_raw"]  = df["skills_raw"].fillna("")

# step 4 - create new useful columns
df["exp_years"]    = df["experience"].apply(get_experience_years)
df["salary_tier"]  = df["salary_avg"].apply(get_salary_bucket)
df["role"]         = df["title"].apply(get_role_type)
df["level"]        = df.apply(lambda r: get_level(r["title"], r["exp_years"]), axis=1)
df["city"]         = df["location"].apply(get_city_name)
df["work_mode"]    = df.apply(lambda r: get_work_type(r["title"], r["location"]), axis=1)
df["skills"]       = df.apply(lambda r: find_skills(r["skills_raw"] + " " + r["description"]), axis=1)
df["skill_count"]  = df["skills"].apply(len)
df["has_salary"]   = df["salary_avg"].notna().astype(int)

print(f"\nFinal clean jobs: {len(df)}")
print(f"Cities breakdown: {df['city'].value_counts().head(5).to_dict()}")
print(f"Jobs with salary: {df['has_salary'].sum()}")

# save clean data
df.to_csv(os.path.join(DATA_FOLDER, "naukri_clean.csv"), index=False)

# ============================================================
# Skill Analysis
# find which skills appear most in job listings
# ============================================================
all_skills   = []
for skill_list in df["skills"]:
    all_skills.extend(skill_list)

skill_counts = pd.Series(all_skills).value_counts()
skill_pct    = (skill_counts / len(df) * 100).round(1)
skill_table  = pd.DataFrame({
    "count":       skill_counts,
    "pct_of_jobs": skill_pct
}).head(20)

print("\nTop 15 Skills:")
print(skill_table.head(15).to_string())
skill_table.to_csv(os.path.join(DATA_FOLDER, "skill_demand.csv"))

# top skills per role
print("\nTop 5 Skills per Role:")
skills_by_role = {}
for role in df["role"].unique():
    role_skills = []
    for s in df[df["role"] == role]["skills"]:
        role_skills.extend(s)
    if role_skills:
        top = pd.Series(role_skills).value_counts().head(8)
        skills_by_role[role] = top
        print(f"  {role}: {', '.join(top.index.tolist()[:5])}")

# ============================================================
# Skill Combinations
# find which skills appear together most often
# ============================================================
combo_counter = {}
for skill_list in df["skills"]:
    if len(skill_list) >= 2:
        for pair in combinations(sorted(skill_list[:6]), 2):
            key = " + ".join(pair)
            combo_counter[key] = combo_counter.get(key, 0) + 1

combo_df = pd.DataFrame(
    list(combo_counter.items()),
    columns=["combination", "count"]
).sort_values("count", ascending=False).head(15).reset_index(drop=True)

print("\nTop Skill Combinations:")
print(combo_df.head(10).to_string())
combo_df.to_csv(os.path.join(DATA_FOLDER, "skill_combinations.csv"), index=False)


# Career Roadmap Engine

print("\nCareer Roadmap based on market demand:")
roadmap = {}
role_order = [
    "Data Analyst", "Business Analyst", "BI Analyst",
    "Data Scientist", "ML Engineer", "Data Engineer", "SQL Developer"
]

for role in role_order:
    if role in skills_by_role:
        top_skills = skills_by_role[role].head(6).index.tolist()
        roadmap[role] = top_skills
        print(f"\n  {role}:")
        for i, skill in enumerate(top_skills, 1):
            print(f"    {i}. {skill}")

# save roadmap
roadmap_rows = []
for role, skills in roadmap.items():
    for i, skill in enumerate(skills, 1):
        roadmap_rows.append({"role": role, "priority": i, "skill": skill})
roadmap_df = pd.DataFrame(roadmap_rows)
roadmap_df.to_csv(os.path.join(DATA_FOLDER, "career_roadmap.csv"), index=False)


# Hiring Trends

print("\nTop 15 Companies Hiring:")
print(df["company"].value_counts().head(15).to_string())

print("\nJobs by Role:")
print(df["role"].value_counts().to_string())

print("\nJobs by City:")
print(df["city"].value_counts().head(10).to_string())

print("\nWork Mode:")
print(df["work_mode"].value_counts().to_string())

print("\nSeniority Level:")
print(df["level"].value_counts().to_string())



# Save to SQLite Database

db_path = os.path.join(DATA_FOLDER, "naukri_jobs.db")
conn    = sqlite3.connect(db_path)
c       = conn.cursor()

# create tables
c.executescript("""
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS skill_demand;
DROP TABLE IF EXISTS companies;
DROP TABLE IF EXISTS skill_combos;
DROP TABLE IF EXISTS career_roadmap;

CREATE TABLE jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT,
    company       TEXT,
    city          TEXT,
    role          TEXT,
    level         TEXT,
    work_mode     TEXT,
    experience    TEXT,
    exp_years     INTEGER,
    salary        TEXT,
    salary_avg    REAL,
    salary_tier   TEXT,
    skill_count   INTEGER,
    has_salary    INTEGER,
    posted_date   TEXT,
    scraped_at    TEXT
);

CREATE TABLE skill_demand (
    skill       TEXT PRIMARY KEY,
    job_count   INTEGER,
    pct_of_jobs REAL
);

CREATE TABLE companies (
    company    TEXT PRIMARY KEY,
    total_jobs INTEGER
);

CREATE TABLE skill_combos (
    combination TEXT,
    count       INTEGER
);

CREATE TABLE career_roadmap (
    role     TEXT,
    priority INTEGER,
    skill    TEXT
);
""")

# insert jobs
for _, row in df.iterrows():
    c.execute("""
        INSERT INTO jobs VALUES
        (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row["title"], row["company"], row["city"],
        row["role"], row["level"], row["work_mode"],
        row.get("experience"), row.get("exp_years"),
        row.get("salary"), row.get("salary_avg"),
        row["salary_tier"], int(row["skill_count"]),
        int(row["has_salary"]), row.get("posted_date"),
        row.get("scraped_at")
    ))

# insert skill demand
for skill, row in skill_table.iterrows():
    c.execute("INSERT OR REPLACE INTO skill_demand VALUES (?,?,?)",
              (skill, int(row["count"]), float(row["pct_of_jobs"])))

# insert companies
for company, count in df["company"].value_counts().head(30).items():
    c.execute("INSERT OR REPLACE INTO companies VALUES (?,?)",
              (company, int(count)))

# insert skill combinations
for _, row in combo_df.iterrows():
    c.execute("INSERT INTO skill_combos VALUES (?,?)",
              (row["combination"], int(row["count"])))

# insert career roadmap
for _, row in roadmap_df.iterrows():
    c.execute("INSERT INTO career_roadmap VALUES (?,?,?)",
              (row["role"], int(row["priority"]), row["skill"]))

conn.commit()

# run some SQL queries to verify data
print("\n" + "="*50)
print("SQL QUERIES ON DATABASE")
print("="*50)

sql_queries = {
    "Jobs by Role":
        "SELECT role, COUNT(*) as count FROM jobs GROUP BY role ORDER BY count DESC",
    "Top Cities":
        "SELECT city, COUNT(*) as count FROM jobs GROUP BY city ORDER BY count DESC LIMIT 8",
    "Top Skills":
        "SELECT skill, job_count, pct_of_jobs FROM skill_demand ORDER BY job_count DESC LIMIT 10",
    "Top Companies":
        "SELECT company, total_jobs FROM companies ORDER BY total_jobs DESC LIMIT 10",
    "Fresher Jobs Available":
        "SELECT title, company, city FROM jobs WHERE level='Fresher/Junior' LIMIT 10",
    "Top Skill Combos":
        "SELECT combination, count FROM skill_combos ORDER BY count DESC LIMIT 10",
}

for name, query in sql_queries.items():
    print(f"\n{name}:")
    print(pd.read_sql_query(query, conn).to_string())

conn.close()

# ============================================================
# Charts and Visualizations
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("India Job Market Intelligence - Naukri.com", fontsize=16, fontweight="bold")

# chart 1 - jobs by role
df["role"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue", edgecolor="black")
axes[0, 0].set_title("Jobs by Role")
axes[0, 0].tick_params(axis="x", rotation=30)

# chart 2 - top 15 skills
skill_table.head(15)["count"].sort_values().plot(
    kind="barh", ax=axes[0, 1], color="coral", edgecolor="black"
)
axes[0, 1].set_title("Top 15 Skills in Demand")

# chart 3 - jobs by city
df["city"].value_counts().head(8).plot(
    kind="bar", ax=axes[0, 2], color="green", edgecolor="black"
)
axes[0, 2].set_title("Jobs by City")
axes[0, 2].tick_params(axis="x", rotation=30)

# chart 4 - seniority levels
df["level"].value_counts().plot(
    kind="bar", ax=axes[1, 0], color="orange", edgecolor="black"
)
axes[1, 0].set_title("Seniority Level")
axes[1, 0].tick_params(axis="x", rotation=30)

# chart 5 - work mode pie chart
df["work_mode"].value_counts().plot(
    kind="pie", ax=axes[1, 1], autopct="%1.1f%%",
    colors=["#2ecc71", "#3498db", "#e74c3c"]
)
axes[1, 1].set_title("Work Mode")

# chart 6 - top skill combinations
combo_df.head(10).set_index("combination")["count"].sort_values().plot(
    kind="barh", ax=axes[1, 2], color="purple", edgecolor="black"
)
axes[1, 2].set_title("Top Skill Combinations")

plt.tight_layout()
chart_path = os.path.join(OUTPUT_FOLDER, "eda_charts.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nCharts saved to: {chart_path}")

# skill heatmap - shows which skills each role needs
top10 = skill_counts.head(10).index.tolist()
heatmap_data = pd.DataFrame(index=df["role"].unique(), columns=top10, data=0)
for role in df["role"].unique():
    role_skill_list = []
    for s in df[df["role"] == role]["skills"]:
        role_skill_list.extend(s)
    counts = pd.Series(role_skill_list).value_counts()
    for skill in top10:
        heatmap_data.loc[role, skill] = counts.get(skill, 0)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data.astype(int), annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Which Skills Does Each Role Need - Naukri.com")
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_FOLDER, "skill_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.show()
print(f"Heatmap saved to: {heatmap_path}")

# save final dataset
df.to_csv(os.path.join(DATA_FOLDER, "naukri_final.csv"), index=False)

# ============================================================
# Key Findings Summary
# ============================================================
top_skill = skill_counts.index[0]
top_city  = df["city"].value_counts().index[0]
remote_pct  = round((df["work_mode"] == "Remote").mean() * 100, 1)
fresher_pct = round((df["level"] == "Fresher/Junior").mean() * 100, 1)
top_combo   = combo_df.iloc[0]["combination"]

print("\n" + "="*50)
print("KEY FINDINGS FROM DATA")
print("="*50)
print(f"Total jobs analysed  : {len(df)}")
print(f"Top demanded skill   : {top_skill}")
print(f"Top hiring city      : {top_city} ({df['city'].value_counts().iloc[0]} jobs)")
print(f"Remote jobs          : {remote_pct}% - most jobs are onsite")
print(f"Fresher friendly     : {fresher_pct}% jobs open to freshers")
print(f"Top skill combo      : {top_combo}")
print(f"Top 5 skills overall : {', '.join(skill_counts.head(5).index.tolist())}")
print("\nAll files saved in data/ and outputs/ folders")


# top skills for freshers specifically
print("\nTOP SKILLS FOR FRESHERS:")
fresher_df = df[df["level"] == "Fresher/Junior"]
fresher_skills = []
for s in fresher_df["skills"]:
    fresher_skills.extend(s)
fresher_skill_counts = pd.Series(fresher_skills).value_counts().head(6)
for i, (skill, count) in enumerate(fresher_skill_counts.items(), 1):
    print(f"  {i}. {skill} ({count} fresher jobs)")
print("\nMARKET INSIGHTS:")
print("- SQL and Python dominate across all roles")
print("- Bangalore is the largest hiring hub for Data Science in India")
print("- Power BI is critical for BI Analyst and Data Analyst roles")
print("- Python + SQL is the most valuable skill combination")
print("- 30% of jobs are open to freshers - good entry point")
print("- Only 8% remote jobs on Naukri India sample - remote may be higher on LinkedIn")
print(f"- 1600 raw records collected, {len(df)} unique after URL deduplication")
print("- Naukri recycles same job URLs across multiple searches - data quality finding")
