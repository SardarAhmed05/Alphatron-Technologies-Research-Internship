"""
Unit tests for WebScraperEngine and DOM Sanitizer.
"""

import os
import pytest
from src.scraper import WebScraperEngine


@pytest.fixture
def scraper():
    return WebScraperEngine()


def test_clean_html_basic(scraper):
    raw_html = "<html><head><title>Test Page</title><script>var x=1;</script></head><body><h1>Main Title</h1><p>Sample content here.</p></body></html>"
    result = scraper.clean_html(raw_html, source_url="https://example.com")
    
    assert result["title"] == "Test Page"
    assert "Main Title" in result["cleaned_content"]
    assert "Sample content here." in result["cleaned_content"]
    assert "var x=1" not in result["cleaned_content"]


def test_table_to_markdown_conversion(scraper):
    raw_html = """
    <table>
        <tr><th>Name</th><th>Price</th></tr>
        <tr><td>Product A</td><td></td></tr>
    </table>
    """
    result = scraper.clean_html(raw_html)
    assert "| Name | Price |" in result["cleaned_content"]
    assert "| Product A |  |" in result["cleaned_content"]


def test_domain_trust_scoring(scraper):
    assert scraper.calculate_domain_trust("https://arxiv.org/abs/1234") >= 95.0
    assert scraper.calculate_domain_trust("https://nature.com/articles") >= 95.0
    assert scraper.calculate_domain_trust("https://random-spam.xyz") <= 60.0
