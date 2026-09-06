from __future__ import annotations

import json

from wmloop.retrieve.providers import search_provider


def test_shared_research_providers_parse_data_only_metadata() -> None:
    atom = b"""<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>https://arxiv.org/abs/2601.00001</id><title>World Model</title>
      <summary>Long horizon rollout stability.</summary><published>2026-01-01</published>
      <link title='pdf' href='https://arxiv.org/pdf/2601.00001'/></entry>
    </feed>"""
    github = json.dumps(
        {
            "items": [
                {
                    "full_name": "org/world-model",
                    "name": "world-model",
                    "description": "Long-horizon video rollout codebase",
                    "html_url": "https://github.com/org/world-model",
                }
            ]
        }
    ).encode()
    openalex = json.dumps(
        {
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "display_name": "Temporal Memory",
                    "abstract_inverted_index": {
                        "Long": [0],
                        "horizon": [1],
                        "memory": [2],
                    },
                    "doi": "https://doi.org/10.1/example",
                    "publication_date": "2026-01-02",
                }
            ]
        }
    ).encode()

    def fetch(url: str, _timeout: float, _limit: int) -> bytes:
        if "arxiv" in url:
            return atom
        if "github" in url:
            return github
        return openalex

    arxiv = search_provider(
        "arxiv", "world model", limit=2, timeout_seconds=1, byte_limit=10_000, fetch=fetch
    )
    github_rows = search_provider(
        "github", "world model", limit=2, timeout_seconds=1, byte_limit=10_000, fetch=fetch
    )
    openalex_rows = search_provider(
        "openalex", "world model", limit=2, timeout_seconds=1, byte_limit=10_000, fetch=fetch
    )
    assert arxiv[0].record_id == "2601.00001"
    assert arxiv[0].source_url.endswith("2601.00001")
    assert github_rows[0].record_id == "org/world-model"
    assert openalex_rows[0].summary == "Long horizon memory"
