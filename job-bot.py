import os
import re
import time
import argparse
import logging
import urllib.parse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# =============================
# LOGGING
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================
# CONFIG
# =============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Recipients: comma-separated env var, fallback to sender only
_raw_recipients = os.getenv("EMAIL_RECIPIENTS", EMAIL_USER or "")
RECIPIENTS: list[str] = [r.strip() for r in _raw_recipients.split(",") if r.strip()]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_URL = "https://infopark.in"

# Max characters sent to Groq in a single call (stays comfortably inside context window)
GROQ_CHAR_BUDGET = 20_000

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

@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _get(url: str, timeout: int = 15) -> requests.Response:
    """HTTP GET with automatic retries on transient failures."""
    logger.debug("GET %s", url)
    return requests.get(url, timeout=timeout)


def fetch_recent_jobs() -> list[dict]:
    recent_jobs: dict[str, dict] = {}
    today = datetime.now().date()
    cutoff = today - timedelta(days=1)

    for keyword in KEYWORDS:
        page = 1

        while True:
            search_term = urllib.parse.quote(keyword)
            url = f"{BASE_URL}/companies/job-search?search={search_term}&page={page}"

            logger.info("Searching '%s' — page %d", keyword, page)

            try:
                r = _get(url)
            except requests.RequestException as exc:
                logger.error("Failed to fetch page for keyword '%s': %s", keyword, exc)
                break

            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("table tbody tr")

            if not rows:
                break  # No more pages

            oldest_date_on_page: datetime.date | None = None

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue

                date_str = cols[0].text.strip()
                job_title = cols[1].text.strip()

                try:
                    job_date = datetime.strptime(date_str, "%d-%m-%Y").date()
                except ValueError:
                    continue

                # Track the oldest date on this page (to decide whether to paginate)
                if oldest_date_on_page is None or job_date < oldest_date_on_page:
                    oldest_date_on_page = job_date

                if job_date < cutoff:
                    continue  # Row is too old, skip but keep scanning this page

                details_cell = cols[-1]
                link_tag = details_cell.find("a")
                if not link_tag:
                    continue

                link: str = link_tag.get("href", "")
                if link and not link.startswith("http"):
                    link = BASE_URL + link

                if link:
                    recent_jobs[link] = {
                        "title": job_title,
                        "date": date_str,
                        "link": link,
                    }

            # Only stop paginating when every job on this page is older than the cutoff.
            # This avoids missing recent jobs on pages that also contain older listings.
            if oldest_date_on_page is not None and oldest_date_on_page < cutoff:
                break

            page += 1

    logger.info("Found %d unique recent jobs.", len(recent_jobs))
    return list(recent_jobs.values())


# =============================
# 2️⃣ EXTRACT JOB DESCRIPTION
# =============================

def extract_description(link: str) -> str | None:
    try:
        r = _get(link)
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator="\n")[:1500]
    except Exception as exc:
        logger.error("Error fetching %s: %s", link, exc)
        return None


# =============================
# 2b️⃣ EXPERIENCE FILTER (≤ 1 YEAR)
# =============================

# Max experience threshold (inclusive)
MAX_EXPERIENCE_YEARS: float = 1.0

def extract_min_experience(text: str) -> float | None:
    """
    Parse the *minimum* years of experience mentioned in a job description.

    Returns:
        float  — minimum years required (e.g. 0.0 for freshers, 1.0, 2.0 …)
        None   — could not determine; caller should NOT filter this job out.

    Handles patterns like:
        fresher / freshers / fresh graduate / no experience
        0-1 year(s) / 0 to 1 year / 1+ year / 1 year / 6 months / 0.5 year
        2-3 years / 3+ years / minimum 2 years etc.
    """
    lower = text.lower()

    # Fresher / no experience → 0 years
    if re.search(
        r"\b(fresher|freshers|fresh\s+graduate|no\s+experience|0\s+years?\s+experience)\b",
        lower,
    ):
        return 0.0

    # "6 months" alone (without a year range) → treat as 0.5 years
    if re.search(r"\b6\s*months?\b", lower) and not re.search(
        r"\d+\s*[-–]\s*\d+\s*years?", lower
    ):
        return 0.5

    # Range: "0-1 year", "1-2 years", "2 - 3 years"
    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?",
        lower,
    )
    if range_match:
        return float(range_match.group(1))  # minimum of the range

    # Single value: "1+ year", "minimum 2 years", "at least 3 years"
    single_match = re.search(
        r"(?:minimum|min\.?|at\s+least|over|above|more\s+than)?\s*"
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?(?:exp|experience)?",
        lower,
    )
    if single_match:
        return float(single_match.group(1))

    return None  # can't determine → do NOT filter out


def is_entry_level(description: str) -> bool:
    """
    Return True if the job requires ≤ MAX_EXPERIENCE_YEARS years experience,
    or if we cannot determine the experience requirement.
    """
    min_exp = extract_min_experience(description)
    if min_exp is None:
        return True  # uncertain → include to avoid missing opportunities
    return min_exp <= MAX_EXPERIENCE_YEARS


# =============================
# 3️⃣ GROQ SUMMARY
# =============================

def summarize(all_jobs_text: str) -> str:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set.")
        return "Groq API key missing — summary unavailable."

    # Enforce character budget
    if len(all_jobs_text) > GROQ_CHAR_BUDGET:
        logger.warning(
            "Combined job text (%d chars) exceeds budget (%d chars). Truncating.",
            len(all_jobs_text),
            GROQ_CHAR_BUDGET,
        )
        all_jobs_text = all_jobs_text[:GROQ_CHAR_BUDGET] + "\n\n[...truncated]"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Below are multiple job postings.\n\n"
                    "For EACH job, extract clearly:\n"
                    "- Company Name\n"
                    "- Job Title\n"
                    "- Required Skills\n"
                    "- Experience\n"
                    "- Location\n"
                    "- Salary (if mentioned)\n\n"
                    "Return clean bullet-point format per job.\n\n"
                    f"{all_jobs_text}"
                ),
            }
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        logger.error("Groq API error (%d): %s", response.status_code, response.text)
        return "Failed to summarize — Groq API returned an error."
    except requests.RequestException as exc:
        logger.error("Groq request failed: %s", exc)
        return "Failed to summarize — network error."


# =============================
# 4️⃣ BUILD EMAIL CONTENT
# =============================

def build_report(jobs: list[dict], summary: str) -> tuple[str, str]:
    """Return (plain_text, html) versions of the report."""
    today_str = datetime.now().strftime("%d %b %Y")

    # ----- PLAIN TEXT -----
    plain = f"🔥 DAILY INFOPARK JOB REPORT — {today_str}\n"
    plain += f"Found {len(jobs)} new job(s) in the last 24 hours.\n"
    plain += "\n" + "=" * 50 + "\n"
    plain += "📋 JOB SUMMARIES\n"
    plain += "=" * 50 + "\n\n"
    plain += summary
    plain += "\n\n" + "=" * 50 + "\n"
    plain += "🔗 JOB LINKS\n"
    plain += "=" * 50 + "\n"
    for job in jobs:
        plain += f"\n• {job['title']} ({job['date']})\n  {job['link']}\n"

    # ----- HTML -----
    links_html = "".join(
        f'<li><a href="{job["link"]}">{job["title"]}</a> '
        f'<span style="color:#888;font-size:0.85em">({job["date"]})</span></li>'
        for job in jobs
    )
    summary_html = summary.replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 20px; }}
    .container {{ max-width: 700px; margin: auto; background: #fff; border-radius: 8px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
    .header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: #fff;
               padding: 24px 32px; }}
    .header h1 {{ margin: 0; font-size: 1.5em; }}
    .header p {{ margin: 6px 0 0; opacity: 0.85; font-size: 0.9em; }}
    .section {{ padding: 24px 32px; border-bottom: 1px solid #e8eaed; }}
    .section h2 {{ margin-top: 0; font-size: 1.1em; color: #1a73e8; }}
    .summary {{ white-space: pre-wrap; font-size: 0.92em; line-height: 1.6; }}
    ul.links {{ padding-left: 20px; }}
    ul.links li {{ margin-bottom: 10px; font-size: 0.92em; }}
    ul.links a {{ color: #1a73e8; text-decoration: none; }}
    ul.links a:hover {{ text-decoration: underline; }}
    .footer {{ padding: 16px 32px; font-size: 0.78em; color: #888; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔥 Daily Infopark Job Report</h1>
      <p>{today_str} &mdash; {len(jobs)} new job(s) in the last 24 hours</p>
    </div>
    <div class="section">
      <h2>📋 Job Summaries</h2>
      <div class="summary">{summary_html}</div>
    </div>
    <div class="section">
      <h2>🔗 Apply Links</h2>
      <ul class="links">{links_html}</ul>
    </div>
    <div class="footer">Powered by Infopark scraper + Groq LLaMA · {today_str}</div>
  </div>
</body>
</html>"""

    return plain, html


def build_no_jobs_report() -> tuple[str, str]:
    today_str = datetime.now().strftime("%d %b %Y")
    plain = f"ℹ️ No new Infopark jobs found for {today_str}. The bot ran successfully."
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;padding:24px;color:#555;">
  <h2 style="color:#1a73e8;">Infopark Job Bot</h2>
  <p>✅ The bot ran successfully on <strong>{today_str}</strong>.</p>
  <p>No new job postings matched your keywords in the last 24 hours.</p>
</body></html>"""
    return plain, html


# =============================
# 5️⃣ SEND EMAIL
# =============================

def send_email(subject: str, plain: str, html: str, dry_run: bool = False) -> None:
    if dry_run:
        logger.info("--- DRY RUN: Email not sent ---")
        print("\n" + "=" * 60)
        print(f"SUBJECT: {subject}")
        print("=" * 60)
        print(plain)
        print("=" * 60 + "\n")
        return

    if not EMAIL_USER or not EMAIL_PASSWORD:
        logger.error("EMAIL_USER or EMAIL_PASSWORD not set — cannot send email.")
        return

    if not RECIPIENTS:
        logger.error("No recipients configured (EMAIL_RECIPIENTS env var is empty).")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = ", ".join(RECIPIENTS)

        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, RECIPIENTS, msg.as_string())

        logger.info("📧 Email sent to: %s", ", ".join(RECIPIENTS))

    except Exception as exc:
        logger.error("Email send failed: %s", exc)


# =============================
# MAIN
# =============================

def main() -> None:
    parser = argparse.ArgumentParser(description="Infopark Daily Job Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and summarise jobs but print report to stdout instead of sending email.",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting Job Bot (dry_run=%s) ...", args.dry_run)

    jobs = fetch_recent_jobs()
    today_str = datetime.now().strftime("%d %b %Y")

    if not jobs:
        logger.warning("No recent jobs found in the last 24 hours.")
        subject = f"ℹ️ Infopark Job Bot: no new jobs ({today_str})"
        plain, html = build_no_jobs_report()
        send_email(subject, plain, html, dry_run=args.dry_run)
        return

    combined_text = ""
    jobs_with_desc: list[dict] = []
    skipped_exp = 0

    for job in jobs:
        logger.info("Processing: %s (%s)", job["title"], job["date"])
        desc = extract_description(job["link"])
        if not desc:
            continue

        # ── Experience filter ──────────────────────────────────────────────
        if not is_entry_level(desc):
            min_exp = extract_min_experience(desc)
            logger.info(
                "⏭️  Skipped (exp > %.0f yr): %s — requires ~%.0f yr",
                MAX_EXPERIENCE_YEARS, job["title"], min_exp or 0,
            )
            skipped_exp += 1
            continue
        # ──────────────────────────────────────────────────────────────────

        combined_text += f"\n\nJOB TITLE: {job['title']}\nPOSTED: {job['date']}\n{desc}"
        jobs_with_desc.append(job)

    logger.info(
        "After experience filter: %d kept, %d skipped (> %.0f yr).",
        len(jobs_with_desc), skipped_exp, MAX_EXPERIENCE_YEARS,
    )

    summary = summarize(combined_text)
    plain, html = build_report(jobs_with_desc, summary)
    subject = f"🔥 Infopark Jobs: {len(jobs_with_desc)} new listing(s) — {today_str}"

    send_email(subject, plain, html, dry_run=args.dry_run)
    logger.info("✅ Job Bot completed successfully.")


if __name__ == "__main__":
    main()
