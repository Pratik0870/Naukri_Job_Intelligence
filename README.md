# India Job Market Intelligence System

A complete end-to-end Data Science project that scrapes live job data from Naukri.com, analyzes skill demand, hiring trends, and builds an interactive Streamlit dashboard.

## Why I Built This

I was job hunting and wanted to understand which skills are actually demanded in India's Data Science market. Instead of guessing, I built a system to scrape and analyze real data from Naukri.com.

## Key Findings

- **Python + SQL** is the most valuable skill combination — appears in 18% of all listings
- **Bangalore** has 42% of all Data Science jobs in India
- Only **8.1%** of jobs are remote — plan to relocate
- **30%** of positions are open to freshers
- **TCS, Accenture, Wipro** are the top hirers

## Data Collection Summary

| Metric | Value |
|---|---|
| Total Records Collected | 1600 |
| Unique Job URLs | 160 |
| Duplicate URLs Removed | 1440 |

**Why so many duplicates?**
Naukri returns the same job posting across multiple search keywords. For example, a "Data Analyst" job at TCS appears in searches for "data analyst", "business analyst", and "analytics engineer". Deduplication was performed using unique job URLs — the most reliable method.

## Project Limitation — Salary Data

Salary information was unavailable for most job postings because employers did not disclose compensation details on Naukri. Therefore salary analytics were excluded from the final analysis. This is a known limitation of Indian job portals — LinkedIn shows more salary data.

## Tech Stack

| Category | Tools |
|---|---|
| Scraping | Selenium, BeautifulSoup |
| Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn (Random Forest) |
| Database | SQLite |
| Dashboard | Streamlit |

## Project Structure

```
naukri-job-intelligence/
├── data/                    # Raw and cleaned CSV files + SQLite DB
├── scraper/
│   └── naukri_scraper.py    # Selenium scraper
├── analysis/
│   └── analysis.py          # EDA + Career Roadmap + SQL pipeline
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── outputs/                 # Charts and visualizations
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Step 1 - Install
pip install -r requirements.txt

# Step 2 - Scrape (LOCAL LAPTOP only - Colab IPs blocked by Naukri)
python scraper/naukri_scraper.py

# Step 3 - Analyse
python analysis/analysis.py

# Step 4 - Dashboard
streamlit run dashboard/app.py
```

## Dashboard Pages

1. **Overview** — KPIs, role distribution, work mode
2. **Skills** — Top skills, heatmap, combinations
3. **Cities** — Geographic analysis
4. **Companies** — Top hirers, company types
5. **Career Roadmap** — Skill recommendations by target role
6. **Job Explorer** — Search and filter, download CSV

## Market Insights

- SQL and Python dominate across all roles
- Bangalore remains the largest hiring hub
- Power BI is critical for BI and Analyst roles
- Python + SQL is the most valuable skill combination
- 30% of jobs are open to freshers — good entry point

## Author

Pratik Patil | B.Sc. Data Science & Business Analytics | Mumbai
GitHub: [github.com/Pratik0870](https://github.com/Pratik0870)
