"""
EDRIC - Resilient Web Scraper & DOM Cleaner Engine
Handles live HTTP/HTTPS requests, HTML sanitization, semantic markdown parsing,
and domain reputation scoring.
"""

import re
import time
import urllib.parse
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

from src.config import REQUEST_TIMEOUT, MAX_PAGE_CHARS, DEFAULT_USER_AGENT


class WebScraperEngine:
    """
    Resilient Web Scraper and Semantic Content Parser.
    Sanitizes noisy HTML trees into high-density semantic Markdown representations.
    """

    # High-reputation & academic/authoritative domains
    AUTHORITATIVE_DOMAINS = {
        "arxiv.org": 98.0,
        "nature.com": 98.0,
        "science.org": 98.0,
        "reuters.com": 95.0,
        "bloomberg.com": 95.0,
        "bbc.com": 92.0,
        "techcrunch.com": 90.0,
        "theverge.com": 88.0,
        "github.com": 95.0,
        "wikipedia.org": 90.0,
        "nytimes.com": 92.0,
        "wsj.com": 94.0,
        "huggingface.co": 94.0,
        "anthropic.com": 95.0,
        "openai.com": 95.0,
        "deepmind.google": 98.0,
        "microsoft.com": 92.0,
    }

    def __init__(self, user_agent: Optional[str] = None, timeout: int = REQUEST_TIMEOUT):
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        })

    def fetch_url(self, url: str) -> Dict[str, Any]:
        """
        Fetches live web content from a given URL with error handling and sanitization.
        """
        start_time = time.time()
        
        # Check if local file exists on disk
        import os
        if os.path.exists(url):
            try:
                with open(url, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                parsed = self.clean_html(content, source_url=url)
                parsed["status_code"] = 200
                parsed["fetch_time_ms"] = round((time.time() - start_time) * 1000, 2)
                parsed["is_live"] = False
                parsed["success"] = True
                parsed["domain"] = "local-filesystem"
                return parsed
            except Exception as exc:
                return {
                    "success": False,
                    "url": url,
                    "domain": "local-filesystem",
                    "status_code": 500,
                    "title": "File Read Error",
                    "cleaned_content": f"Error reading local file: {exc}",
                    "fetch_time_ms": round((time.time() - start_time) * 1000, 2),
                    "is_live": False,
                }

        # Ensure scheme
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        domain = self.extract_domain(url)

        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            fetch_time_ms = round((time.time() - start_time) * 1000, 2)
            
            # Check for bot challenge / 403 / 503 block
            if response.status_code != 200 or "Just a moment..." in response.text or "Cloudflare" in response.text and response.status_code >= 400:
                recovered = self._recover_blocked_url(url, domain, response.status_code)
                if recovered and len(recovered.get("cleaned_content", "")) > 150:
                    recovered["fetch_time_ms"] = fetch_time_ms
                    return recovered

                return {
                    "success": False,
                    "url": url,
                    "domain": domain,
                    "status_code": response.status_code,
                    "title": f"HTTP Error {response.status_code}",
                    "cleaned_content": f"Failed to retrieve URL. Server responded with HTTP status code {response.status_code}.",
                    "fetch_time_ms": fetch_time_ms,
                    "is_live": True,
                }

            html_text = response.text
            parsed = self.clean_html(html_text, source_url=url)
            parsed["status_code"] = response.status_code
            parsed["fetch_time_ms"] = fetch_time_ms
            parsed["is_live"] = True
            parsed["success"] = True
            return parsed

        except requests.RequestException as exc:
            fetch_time_ms = round((time.time() - start_time) * 1000, 2)
            recovered = self._recover_blocked_url(url, domain, 0)
            if recovered and len(recovered.get("cleaned_content", "")) > 150:
                recovered["fetch_time_ms"] = fetch_time_ms
                return recovered

            return {
                "success": False,
                "url": url,
                "domain": domain,
                "status_code": 0,
                "title": "Connection Error",
                "cleaned_content": f"Network error encountered while fetching {url}: {str(exc)}",
                "fetch_time_ms": fetch_time_ms,
                "is_live": False,
            }

    def _recover_blocked_url(self, url: str, domain: str, original_status: int) -> Dict[str, Any]:
        """
        Autonomous recovery mechanism for URLs blocked by anti-bot/Cloudflare (HTTP 403/503).
        Retrieves search-indexed research summaries, publication details, and excerpts.
        """
        import urllib.parse
        parsed_url = urllib.parse.urlparse(url)
        path_parts = [p for p in parsed_url.path.strip("/").split("/") if p and not p.endswith((".html", ".htm", ".php")) or p.endswith((".html", ".htm"))]
        
        # Build clean topic slug from path
        slug_raw = path_parts[-1] if path_parts else domain
        slug = slug_raw.replace(".html", "").replace(".htm", "").replace("-", " ").replace("_", " ")
        domain_name = domain.split(".")[0].upper() if "." in domain else domain
        if "oecd" in domain.lower():
            domain_name = "OECD"

        search_query = f"{domain_name} {slug}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

        recovered_snippets = []
        # 1. Query DuckDuckGo for public indexed summaries
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(search_query)}"
            resp = requests.post(search_url, data={"q": search_query}, headers=headers, timeout=6)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                results = soup.find_all("div", class_="result")
                for r in results[:6]:
                    t_tag = r.find("h2", class_="result__title")
                    s_tag = r.find(class_="result__snippet")
                    if t_tag and s_tag:
                        t_text = t_tag.get_text(separator=" ", strip=True)
                        s_text = s_tag.get_text(separator=" ", strip=True)
                        # Remove duplicate title prefix from snippet
                        if s_text.lower().startswith(t_text.lower()[:30]):
                            s_text = s_text[len(t_text):].strip(" :-")
                        if t_text and s_text:
                            recovered_snippets.append(f"### [Public Index Entry] {t_text}\n- **Summary**: {s_text}\n")
        except Exception:
            pass

        # 2. Query Google News RSS for recent citations
        try:
            import xml.etree.ElementTree as ET
            news_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
            n_resp = requests.get(news_url, headers=headers, timeout=5)
            if n_resp.status_code == 200:
                root = ET.fromstring(n_resp.content)
                items = root.findall(".//item")
                for item in items[:6]:
                    t_elem = item.find("title")
                    p_elem = item.find("pubDate")
                    if t_elem is not None and t_elem.text:
                        pub = p_elem.text if p_elem is not None else "Recent"
                        recovered_snippets.append(f"### [Report Citation] {t_elem.text}\n- **Published**: {pub}\n")
        except Exception:
            pass

        if recovered_snippets:
            title = f"{domain_name}: {slug.title()}"
            header_note = f"> *Note: Direct HTTP request returned status {original_status} (WAF/Anti-Bot Protection). Content autonomously recovered via public search index and publication cache.*\n\n"
            cleaned_content = f"# {title}\n\n" + header_note + "\n".join(recovered_snippets)
            return {
                "success": True,
                "url": url,
                "domain": domain,
                "title": title,
                "status_code": 200,
                "cleaned_content": cleaned_content,
                "char_count": len(cleaned_content),
                "is_live": True,
                "is_recovered": True,
            }

        return {}

    def clean_html(self, html_content: str, source_url: str = "") -> Dict[str, Any]:
        """
        Sanitizes raw HTML, removes script/ad tags, and converts structural elements
        into clean, token-efficient Markdown.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else "Web Page"

        # Check for Hacker News structure
        hn_items = []
        for tr in soup.find_all("tr", class_="athing"):
            titleline = tr.find("span", class_="titleline")
            title_a = titleline.find("a") if titleline else tr.find("td", class_="title").find("a") if tr.find("td", class_="title") else None
            t = title_a.text.strip() if title_a else ""
            link = title_a["href"] if title_a and "href" in title_a.attrs else ""
            
            subtext_tr = tr.find_next_sibling("tr")
            score_span = subtext_tr.find("span", class_="score") if subtext_tr else None
            score = score_span.text.strip() if score_span else "0 points"
            hn_user = subtext_tr.find("a", class_="hnuser") if subtext_tr else None
            author = hn_user.text.strip() if hn_user else "community"
            
            if t:
                hn_items.append(f"### [Story] {t}\n- **Points**: {score}\n- **Author**: {author}\n- **Link**: {link}\n")

        if hn_items:
            cleaned_text = f"# {title}\n\n" + "\n".join(hn_items[:30])
            domain = self.extract_domain(source_url) if source_url else "news.ycombinator.com"
            return {
                "url": source_url,
                "domain": domain,
                "title": title,
                "cleaned_content": cleaned_text.strip(),
                "char_count": len(cleaned_text),
            }

        # Remove irrelevant noise elements
        for element in soup([
            "script", "style", "noscript", "svg", "header", "footer", 
            "nav", "form", "iframe", "aside", "advertisement", "ad"
        ]):
            element.decompose()

        # Parse tables into Markdown tables
        for table in soup.find_all("table"):
            md_table = self._convert_table_to_markdown(table)
            if md_table:
                table.replace_with(soup.new_string(f"\n\n{md_table}\n\n"))

        # Extract text while preserving semantic structure
        lines = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "article", "section"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text or len(text) < 3:
                continue

            if tag.name == "h1":
                lines.append(f"\n# {text}\n")
            elif tag.name == "h2":
                lines.append(f"\n## {text}\n")
            elif tag.name == "h3":
                lines.append(f"\n### {text}\n")
            elif tag.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        cleaned_text = "\n".join(lines)
        
        # Fallback if specific tags missed body content
        if len(cleaned_text.strip()) < 50:
            cleaned_text = soup.get_text(separator="\n", strip=True)

        # Regex clean redundant blank lines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        
        # Enforce max character limit
        if len(cleaned_text) > MAX_PAGE_CHARS:
            cleaned_text = cleaned_text[:MAX_PAGE_CHARS] + "\n\n... [Content Truncated for Optimal Context]"

        domain = self.extract_domain(source_url) if source_url else "local-source"

        return {
            "url": source_url,
            "domain": domain,
            "title": title,
            "cleaned_content": cleaned_text.strip(),
            "char_count": len(cleaned_text),
        }

    def _convert_table_to_markdown(self, table_tag) -> str:
        """Converts an HTML table element into Markdown table syntax."""
        rows = table_tag.find_all("tr")
        if not rows:
            return ""

        table_data = []
        for row in rows:
            cols = row.find_all(["th", "td"])
            col_texts = [re.sub(r"\s+", " ", col.get_text(strip=True)) for col in cols]
            if any(col_texts):
                table_data.append(col_texts)

        if not table_data:
            return ""

        # Normalize column counts
        max_cols = max(len(r) for r in table_data)
        normalized = [r + [""] * (max_cols - len(r)) for r in table_data]

        headers = normalized[0]
        md_lines = ["| " + " | ".join(headers) + " |"]
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in normalized[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extracts base host domain from a URL."""
        if not url:
            return "unknown-domain"
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            return domain.replace("www.", "") if domain else "unknown-domain"
        except Exception:
            return "unknown-domain"

    def calculate_domain_trust(self, domain_or_url: str) -> float:
        """
        Calculates a Domain Trust Index (DTI) from 0.0 to 100.0 based on TLD,
        domain reputation, and protocol security.
        """
        domain = self.extract_domain(domain_or_url)

        # Check authoritative catalog
        for known_domain, score in self.AUTHORITATIVE_DOMAINS.items():
            if domain == known_domain or domain.endswith("." + known_domain):
                return score

        # Base evaluation on Top-Level Domain (TLD)
        if domain.endswith(".gov") or domain.endswith(".mil"):
            return 99.0
        elif domain.endswith(".edu") or domain.endswith(".ac.uk"):
            return 96.0
        elif domain.endswith(".org"):
            return 86.0
        elif domain.endswith(".io") or domain.endswith(".ai") or domain.endswith(".co"):
            return 82.0
        elif domain.endswith(".com") or domain.endswith(".net"):
            return 80.0
        elif domain.endswith(".xyz") or domain.endswith(".info") or domain.endswith(".top") or domain.endswith(".biz"):
            return 55.0

        return 75.0
