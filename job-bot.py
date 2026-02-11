import os
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email.mime.text import MIMEText
import smtplib

load_dotenv()

# =============================
# CONFIG
# =============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

BASE_URL = "https://infopark.in"

KEYWORDS = ["Developer", "Data Analyst", "Python", "AI"]
MAX_PER_KEYWORD = 2  # keep small for free tier


# =============================
# 1️⃣ FETCH JOB LINKS
# =============================

def fetch_jobs():
    job_links = set()

    for keyword in KEYWORDS:
        search_term = urllib.parse.quote(keyword)
        url = f"{BASE_URL}/companies/job-search?search={search_term}"

        print(f"Searching for: {keyword}")
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        links = soup.select("a[href*='/company-jobs/']")

        for a in links[:MAX_PER_KEYWORD]:
            link = a["href"]
            if not link.startswith("http"):
                link = BASE_URL + link
            job_links.add(link)

    return list(job_links)


# =============================
# 2️⃣ EXTRACT DESCRIPTION
# =============================

def extract_description(link):
    try:
        r = requests.get(link, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator="\n")[:2000]  # reduce token usage

    except Exception as e:
        print(f"Error fetching {link}: {e}")
        return None


# =============================
# 3️⃣ SUMMARIZE WITH GROQ
# =============================

def summarize(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": f"""
Extract clearly in bullet points:

- Company Name
- Job Title
- Required Skills
- Experience
- Location
- Salary (if mentioned)

{text}
"""
            }
        ],
        "temperature": 0.2
    }

    for attempt in range(3):
        response = requests.post(GROQ_API_URL, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        if response.status_code == 429:
            print("Rate limited. Waiting 5 seconds...")
            time.sleep(5)
        else:
            print("Groq Error:", response.text)
            return "Failed to summarize."

    return "Skipped due to repeated rate limit."


# =============================
# 4️⃣ SEND EMAIL
# =============================

def send_email(content):
    try:
        msg = MIMEText(content)
        msg["Subject"] = "🔥 Daily Infopark Job Report"
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_USER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("📧 Email sent successfully!")

    except Exception as e:
        print("Email Error:", e)


# =============================
# MAIN
# =============================

def main():
    print("🚀 Starting Job Bot...")

    job_pages = fetch_jobs()

    if not job_pages:
        print("No job pages found.")
        return

    report = "🔥 DAILY INFOPARK JOB REPORT\n"

    for link in job_pages:
        print(f"Processing: {link}")
        desc = extract_description(link)

        if not desc:
            continue

        summary = summarize(desc)

        report += "\n==============================\n"
        report += f"LINK: {link}\n{summary}\n"

        time.sleep(3)  # delay between calls

    send_email(report)
    print("✅ Script completed.")


if __name__ == "__main__":
    main()
