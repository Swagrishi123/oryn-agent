"""Autonomous Web Research Engine: Dual-Layer Wikipedia REST + Google News RSS."""
import re
import urllib.parse
import xml.etree.ElementTree as ET
import requests
from oryn.models.schemas import PermissionCategory, ToolResult
from oryn.tools.registry import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search factual encyclopedic entries and recent news records."
    category = PermissionCategory.READ_ONLY

    def _extract_core_entity(self, query: str) -> str:
        clean = re.sub(
            r'(?i)\b(who is the founder of|who founded|founder of|who is the ceo of|ceo of|'
            r'who is the president of|who is the chairman of|who is|who was|what is|what was|tell me about)\b',
            '',
            query
        )
        return re.sub(r'[,?!\'\"]', '', clean).strip()

    def execute(self, query: str) -> ToolResult:
        clean_entity = self._extract_core_entity(query)
        findings = []

        # 1. Encyclopedic Lookup via Wikipedia REST Summary
        for target in [clean_entity, query.strip()]:
            if not target:
                continue
            try:
                encoded = urllib.parse.quote_plus(target)
                s_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&utf8=&format=json&srlimit=1"
                res = requests.get(s_url, headers={"User-Agent": "ORYNAnalyst/2.0"}, timeout=2.5)
                if res.status_code == 200:
                    results = res.json().get("query", {}).get("search", [])
                    if results and "title" in results[0]:
                        title = results[0]["title"]
                        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                        s_res = requests.get(sum_url, headers={"User-Agent": "ORYNAnalyst/2.0"}, timeout=2.5)
                        if s_res.status_code == 200:
                            extract = s_res.json().get("extract")
                            if extract and len(extract) > 40:
                                findings.append(f"• [{title} Encyclopedic Record]: {extract}")
                                break
            except Exception:
                pass

        # 2. News RSS Feed Lookup for Recent Events & Incumbents
        try:
            encoded_q = urllib.parse.quote_plus(query.strip())
            rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
            res = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.0)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall("./channel/item")[:3]:
                    title = item.findtext("title", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()[:16]
                    clean_title = re.sub(r' - [^-]+$', '', title)
                    if clean_title:
                        findings.append(f"• [{pub_date}] {clean_title}")
        except Exception:
            pass

        if findings:
            return ToolResult(success=True, data="\n\n".join(findings), provenance="web_search", is_live=True)

        return ToolResult(success=False, data=None, error=f"No verified research records found for: {query}")
