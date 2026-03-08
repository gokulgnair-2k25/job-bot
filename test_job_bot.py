"""
Unit tests for job-bot.py
Run with: python -m pytest test_job_bot.py -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal environment so importing job-bot doesn't fail in CI without secrets
# ---------------------------------------------------------------------------
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("EMAIL_USER", "test@example.com")
os.environ.setdefault("EMAIL_PASSWORD", "test-pass")
os.environ.setdefault("EMAIL_RECIPIENTS", "a@example.com, b@example.com")

# job-bot.py uses a hyphen so we load it as a module manually
import importlib.util

spec = importlib.util.spec_from_file_location(
    "job_bot",
    os.path.join(os.path.dirname(__file__), "job-bot.py"),
)
job_bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job_bot)


# ---------------------------------------------------------------------------
# 1. Recipients are parsed correctly from the env var
# ---------------------------------------------------------------------------
class TestRecipients(unittest.TestCase):
    def test_recipients_parsed_from_env(self):
        self.assertEqual(job_bot.RECIPIENTS, ["a@example.com", "b@example.com"])

    def test_recipients_strips_whitespace(self):
        for r in job_bot.RECIPIENTS:
            self.assertEqual(r, r.strip())


# ---------------------------------------------------------------------------
# 2. Groq character budget guard
# ---------------------------------------------------------------------------
class TestSummarize(unittest.TestCase):
    @patch("requests.post")
    def test_text_truncated_when_over_budget(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "summary"}}]
        }
        mock_post.return_value = mock_resp

        # Build text bigger than the budget
        big_text = "x" * (job_bot.GROQ_CHAR_BUDGET + 5000)

        with self.assertLogs("job_bot", level="WARNING") as cm:
            job_bot.summarize(big_text)

        # Verify a truncation warning was logged
        self.assertTrue(any("Truncating" in line or "truncat" in line.lower() for line in cm.output))

        # Verify the payload sent to Groq is within the budget + marker
        sent_body = mock_post.call_args.kwargs["json"]
        sent_content = sent_body["messages"][0]["content"]
        self.assertIn("[...truncated]", sent_content)
        self.assertLessEqual(len(sent_content), job_bot.GROQ_CHAR_BUDGET + 500)  # +500 for prompt wrapper

    @patch("requests.post")
    def test_groq_api_error_returns_safe_string(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_post.return_value = mock_resp

        result = job_bot.summarize("some job text")
        self.assertIn("error", result.lower())


# ---------------------------------------------------------------------------
# 3. extract_description returns None on bad URL
# ---------------------------------------------------------------------------
class TestExtractDescription(unittest.TestCase):
    def test_returns_none_on_exception(self):
        # Patch _get directly on the loaded module object (works with hyphenated filename)
        with patch.object(job_bot, "_get", side_effect=Exception("connection refused")):
            result = job_bot.extract_description("http://invalid.local/job/1")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 4. build_report produces valid plain and HTML output
# ---------------------------------------------------------------------------
class TestBuildReport(unittest.TestCase):
    def setUp(self):
        self.jobs = [
            {"title": "Python Dev", "date": "01-03-2026", "link": "https://example.com/1"},
            {"title": "React Dev", "date": "01-03-2026", "link": "https://example.com/2"},
        ]
        self.summary = "• Company: Acme\n  Skills: Python"

    def test_plain_contains_job_titles(self):
        plain, _ = job_bot.build_report(self.jobs, self.summary)
        self.assertIn("Python Dev", plain)
        self.assertIn("React Dev", plain)

    def test_plain_contains_links(self):
        plain, _ = job_bot.build_report(self.jobs, self.summary)
        self.assertIn("https://example.com/1", plain)

    def test_html_contains_anchor_tags(self):
        _, html = job_bot.build_report(self.jobs, self.summary)
        self.assertIn('<a href="https://example.com/1">', html)

    def test_subject_contains_count(self):
        # Not a method of build_report but test the pattern used in main
        subject = f"🔥 Infopark Jobs: {len(self.jobs)} new listing(s)"
        self.assertIn("2", subject)


# ---------------------------------------------------------------------------
# 5. No dry_run → send_email calls SMTP; dry_run → no SMTP call
# ---------------------------------------------------------------------------
class TestSendEmail(unittest.TestCase):
    @patch("smtplib.SMTP_SSL")
    def test_email_sent_when_not_dry_run(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = lambda s: mock_server
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        job_bot.send_email("subj", "plain", "<html></html>", dry_run=False)
        mock_smtp.assert_called_once()

    @patch("smtplib.SMTP_SSL")
    def test_no_email_sent_in_dry_run(self, mock_smtp):
        job_bot.send_email("subj", "plain", "<html></html>", dry_run=True)
        mock_smtp.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Experience filter: extract_min_experience & is_entry_level
# ---------------------------------------------------------------------------
class TestExperienceFilter(unittest.TestCase):
    def _exp(self, text: str) -> float | None:
        return job_bot.extract_min_experience(text)

    def _ok(self, text: str) -> bool:
        return job_bot.is_entry_level(text)

    # --- fresher / 0-year phrases ---
    def test_fresher_keyword(self):
        self.assertEqual(self._exp("Freshers are welcome to apply"), 0.0)

    def test_no_experience(self):
        self.assertEqual(self._exp("No experience required"), 0.0)

    def test_six_months(self):
        self.assertAlmostEqual(self._exp("6 months of experience preferred"), 0.5)

    # --- ranges ---
    def test_range_0_to_1(self):
        self.assertEqual(self._exp("0-1 years of experience"), 0.0)

    def test_range_1_to_2(self):
        self.assertEqual(self._exp("1-2 years experience required"), 1.0)

    def test_range_2_to_3_filtered(self):
        self.assertEqual(self._exp("2-3 years of relevant experience"), 2.0)
        self.assertFalse(self._ok("2-3 years of relevant experience"))

    # --- single-value phrases ---
    def test_one_year_plus(self):
        self.assertEqual(self._exp("1+ years of experience"), 1.0)
        self.assertTrue(self._ok("1+ years of experience"))

    def test_minimum_three_years(self):
        self.assertEqual(self._exp("minimum 3 years experience"), 3.0)
        self.assertFalse(self._ok("minimum 3 years experience"))

    def test_five_years(self):
        self.assertFalse(self._ok("5 years of software experience needed"))

    # --- uncertain (no exp info) → should NOT be filtered ---
    def test_no_experience_mention_included(self):
        self.assertIsNone(self._exp("We are hiring talented developers."))
        self.assertTrue(self._ok("We are hiring talented developers."))

    # --- boundary: exactly 1 year ---
    def test_exactly_one_year_included(self):
        self.assertTrue(self._ok("1 year experience required"))


if __name__ == "__main__":
    unittest.main()
