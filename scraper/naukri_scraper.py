
# Naukri Job Scraper
# This code opens Chrome automatically and scrapes job data
# from Naukri.com for 8 different Data Science roles

import os
import re
import time
import logging
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# setup logging so we can see what's happening in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")]
)
logger = logging.getLogger(__name__)

# job roles i  want to scrape
JOB_ROLES = [
    "data analyst",
    "data scientist",
    "data engineer",
    "machine learning engineer",
    "business analyst",
    "power bi analyst",
    "sql developer",
    "analytics engineer",
]

PAGES = int(os.getenv("PAGES", 10))


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER    = os.path.join(PROJECT_FOLDER, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)


def open_chrome():
    # setup Chrome so it looks like a real human browser
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def make_naukri_url(role, page):
    # example: https://www.naukri.com/data-analyst-jobs?k=data+analyst&page=2
    slug   = role.strip().lower().replace(" ", "-")
    search = role.strip().replace(" ", "+")
    return f"https://www.naukri.com/{slug}-jobs?k={search}&page={page}"


def get_salary_numbers(salary_text):
    # try to extract salary numbers from text like "3-5 LPA" or "10 LPA"
    # most Naukri jobs hide salary so this will return None most of the time
    if not salary_text or pd.isna(salary_text):
        return None, None, None
    text = str(salary_text).lower().replace(",", "")
    if any(word in text for word in ["not", "disclose", "confidential"]):
        return None, None, None
    # match range like "3-5 lpa"
    match = re.search(r"(\d+\.?\d*)\s*[-to]\s*(\d+\.?\d*)\s*(?:lpa|lac)?", text)
    if match:
        low  = float(match.group(1))
        high = float(match.group(2))
        # convert if salary is in full rupees instead of lakhs
        if low  > 100: low  = round(low  / 100000, 1)
        if high > 100: high = round(high / 100000, 1)
        return low, high, round((low + high) / 2, 1)
    return None, None, None


def read_one_job_card(card, role):
    # extract job details from one job card on the page
    try:
        # job title link
        title_tag = (
            card.find("a", class_=lambda x: x and "title" in str(x).lower()) or
            card.find("a", attrs={"class": "title"})
        )
        # company name
        company_tag = (
            card.find("a", class_=lambda x: x and "comp-name" in str(x).lower()) or
            card.find("a", class_=lambda x: x and "comp" in str(x).lower())
        )
        # experience required
        exp_tag = (
            card.find("span", class_=lambda x: x and "expwdth" in str(x)) or
            card.find("li", class_=lambda x: x and "exp" in str(x).lower())
        )
        # salary (most jobs hide this)
        sal_tag = (
            card.find("span", class_=lambda x: x and "salary" in str(x).lower()) or
            card.find("li", class_=lambda x: x and "sal" in str(x).lower())
        )
        # location
        loc_tag = (
            card.find("span", class_=lambda x: x and "locWdth" in str(x)) or
            card.find("li", class_=lambda x: x and "loc" in str(x).lower())
        )
        # skills shown on card
        skill_tags = card.find_all("li", class_=lambda x: x and "tag" in str(x).lower())
        if not skill_tags:
            skill_tags = card.find_all("span", class_=lambda x: x and "tag" in str(x).lower())

        # short job description preview
        desc_tag   = card.find("div", class_=lambda x: x and "job-desc" in str(x).lower())
        date_tag   = card.find("span", class_=lambda x: x and "job-post-day" in str(x).lower())
        rating_tag = card.find("a", class_=lambda x: x and "rating" in str(x).lower())

        title = title_tag.get_text(strip=True) if title_tag else None
        # skip cards with no title
        if not title or len(title) < 3:
            return None

        salary_text = sal_tag.get_text(strip=True) if sal_tag else None
        sal_min, sal_max, sal_avg = get_salary_numbers(salary_text)

        return {
            "source":      "Naukri",
            "role_search": role,
            "title":       title,
            "company":     company_tag.get_text(strip=True) if company_tag else None,
            "experience":  exp_tag.get_text(strip=True) if exp_tag else None,
            "salary":      salary_text,
            "salary_min":  sal_min,
            "salary_max":  sal_max,
            "salary_avg":  sal_avg,
            "location":    loc_tag.get_text(strip=True) if loc_tag else None,
            "skills_raw":  " | ".join([t.get_text(strip=True) for t in skill_tags[:10]]),
            "description": desc_tag.get_text(strip=True)[:500] if desc_tag else "",
            "posted_date": date_tag.get_text(strip=True) if date_tag else None,
            "company_rating": rating_tag.get_text(strip=True) if rating_tag else None,
            "job_url":     title_tag.get("href", "") if title_tag else None,
            "scraped_at":  datetime.now().isoformat(),
        }
    except Exception:
        return None


def scrape_one_role(driver, role, pages=10):
    # scrape all pages for one job role
    jobs = []
    for page in range(1, pages + 1):
        url = make_naukri_url(role, page)
        try:
            driver.get(url)

            # wait for job cards to appear on page
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "article.jobTuple, div.jobTuple, article")
                    )
                )
            except Exception:
                pass

            # scroll down slowly like a human would
            # this loads lazy-loaded job cards
            time.sleep(4)
            driver.execute_script("window.scrollTo(0, 2000)")
            time.sleep(1.5)
            driver.execute_script("window.scrollTo(0, 4000)")
            time.sleep(1.5)
            driver.execute_script("window.scrollTo(0, 6000)")
            time.sleep(1)

            # read the page HTML after scrolling
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # find all job cards - try different selectors as Naukri changes them
            cards = soup.find_all("article", class_=lambda x: x and "jobTuple" in str(x))
            if not cards:
                cards = soup.find_all("div", class_=lambda x: x and "jobTuple" in str(x))
            if not cards:
                cards = soup.find_all("article", class_=lambda x: x and "job-tuple" in str(x))
            if not cards:
                cards = soup.find_all("article")
            if not cards:
                cards = soup.find_all("div", class_=lambda x: x and "srp-jobtuple" in str(x).lower())

            logger.info(f"{role} | page {page}: {len(cards)} cards found")
            if not cards:
                break

            # read each job card
            page_jobs = 0
            for card in cards:
                job = read_one_job_card(card, role)
                if job:
                    jobs.append(job)
                    page_jobs += 1

            logger.info(f"  Extracted: {page_jobs} jobs")
            if page_jobs == 0:
                break

            
            time.sleep(3)

        except Exception as e:
            logger.warning(f"Error on page {page}: {e}")
            break

    logger.info(f"Total for '{role}': {len(jobs)} jobs")
    return jobs


def run_scraper():
    print("Opening Chrome browser.")
    driver  = open_chrome()
    all_jobs = []

    try:
        for role in JOB_ROLES:
            print(f"\nScraping: {role}")
            jobs = scrape_one_role(driver, role, pages=PAGES)
            all_jobs.extend(jobs)
            # wait between roles
            time.sleep(5)
    finally:
        driver.quit()
        print("Chrome closed.")

    # save to CSV
    df        = pd.DataFrame(all_jobs)
    save_path = os.path.join(DATA_FOLDER, "naukri_raw.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved {len(df)} jobs to: {save_path}")
    return df


if __name__ == "__main__":
    raw_path = os.path.join(DATA_FOLDER, "naukri_raw.csv")

    # if we already scraped before, load existing data
    # this way we don't have to scrape again every time
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 500:
        df = pd.read_csv(raw_path)
        print(f"Loaded existing data: {len(df)} jobs from {raw_path}")
    else:
        df = run_scraper()

    print(f"\nTotal jobs scraped: {len(df)}")
    print(df["role_search"].value_counts())
