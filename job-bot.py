import os
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email.mime.text import MIMEText
import smtplib
from datetime import datetime, timedelta

load_dotenv()

# =============================
# CONFIG
# =============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_URL = "https://infopark.in"

KEYWORDS = [
    # AI & ML
    "AI",
    "Artificial Intelligence",
    "Machine Learning",
    
    "Data ",


    # Developer Roles
    "Developer",
    "Software Developer",
    "Software Engineer",
    "Full Stack",
    "Backend",
    "Frontend",

    # Tech Stack
    "Python",
    
    "Node",
    ".NET",
    "Dotnet",
    "React",
    
    
    

   
]



# =============================
# 1️⃣ FETCH RECENT JOBS (LAST 1 DAY)
# =============================

def fetch_recent_jobs():
    recent_jobs = {}
    today = datetime.now().date()
    one_day_ago = today - timedelta(days=1)

    for keyword in KEYWORDS:
        page = 1

        while True:
            search_term = urllib.parse.quote(keyword)
            url = f"{BASE_URL}/companies/job-search?search={search_term}&page={page}"

            print(f"Searching '{keyword}' - Page {page}")

            r = requests.get(url, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")

            rows = soup.select("table tbody tr")

            if not rows:
                break  # No more pages

            page_has_recent_job = False

            for row in rows:
                cols = row.find_all("td")

                if len(cols) < 5:
                    continue

                date_str = cols[0].text.strip()
                job_title = cols[1].text.strip()

                try:
                    job_date = datetime.strptime(date_str, "%d-%m-%Y").date()
                except:
                    continue

                # Only keep last 1 day jobs
                if job_date < one_day_ago:
                    continue

                page_has_recent_job = True

                # Get details link
                details_cell = cols[-1]
                link_tag = details_cell.find("a")

                if not link_tag:
                    continue

                link = link_tag.get("href")
                if link and not link.startswith("http"):
                    link = BASE_URL + link

                recent_jobs[link] = {
                    "title": job_title,
                    "date": date_str,
                    "link": link
                }

            # If this page had no recent jobs → stop scanning more pages
            if not page_has_recent_job:
                break

            page += 1

    print(f"Found {len(recent_jobs)} recent jobs.")
    return list(recent_jobs.values())


# =============================
# 2️⃣ EXTRACT JOB DESCRIPTION
# =============================

def extract_description(link):
    try:
        r = requests.get(link, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        return soup.get_text(separator="\n")[:1500]

    except Exception as e:
        print(f"Error fetching {link}: {e}")
        return None


# =============================
# 3️⃣ SINGLE GROQ SUMMARY
# =============================

def summarize(all_jobs_text):
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
Below are multiple job postings.

For EACH job, extract clearly:

- Company Name
- Job Title
- Required Skills
- Experience
- Location
- Salary (if mentioned)

Return clean bullet-point format per job.

{all_jobs_text}
"""
            }
        ],
        "temperature": 0.2
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    print("Groq Error:", response.text)
    return "Failed to summarize."


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

    jobs = fetch_recent_jobs()

    if not jobs:
        print("No recent jobs found (last 1 day).")
        return

    combined_text = ""
    link_section = ""

    for job in jobs:
        print(f"Processing: {job['title']} ({job['date']})")

        desc = extract_description(job["link"])
        if not desc:
            continue

        combined_text += f"\n\nJOB TITLE: {job['title']}\nPOSTED: {job['date']}\n{desc}"

        link_section += f"\n🔗 {job['title']} ({job['date']})\n{job['link']}\n"

    # Single Groq API call
    summary = summarize(combined_text)

    report = "🔥 DAILY INFOPARK JOB REPORT (Last 1 Day)\n"
    report += "\n==============================\n"
    report += summary
    report += "\n\n==============================\n"
    report += "📎 JOB LINKS\n"
    report += link_section

    send_email(report)
    print("✅ Script completed successfully!")


if __name__ == "__main__":
    main()
