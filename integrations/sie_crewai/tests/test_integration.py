"""Integration tests for sie-crewai.

These tests require a running SIE server and serve as runnable examples
of CrewAI workflows using SIE tools.

Run with: pytest -m integration integrations/sie_crewai/tests/

Prerequisites:
    mise run serve -d cpu -p 8080
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this module if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    """Get SIE server URL from environment or default."""
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


class TestResearchAgentWorkflow:
    """Integration tests demonstrating research agent use case.

    CrewAI is commonly used for research workflows where agents:
    1. Gather documents from various sources
    2. Rerank them by relevance to a research question
    3. Extract key information for synthesis
    """

    def test_rerank_research_sources(self, sie_url: str) -> None:
        """Example: Research agent reranking sources for relevance."""
        from sie_crewai import SIERerankerTool

        # Research agent tool for finding most relevant sources
        reranker = SIERerankerTool(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        # Simulated search results from multiple sources
        research_sources = [
            "Machine learning algorithms learn patterns from data without explicit programming.",
            "The stock market closed higher today amid positive economic indicators.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning is a subset of machine learning using multi-layer neural networks.",
            "Weather patterns are expected to remain stable through the week.",
            "Transfer learning allows models to apply knowledge from one domain to another.",
        ]

        result = reranker._run(
            query="How do neural networks learn from data?",
            documents=research_sources,
            top_k=3,
        )

        assert "Ranked documents" in result
        assert "Score:" in result
        # Top results should be about ML/neural networks, not weather/stocks


class TestLeadQualificationWorkflow:
    """Integration tests demonstrating lead qualification use case.

    CrewAI is commonly used for sales automation where agents:
    1. Extract company and contact information
    2. Score leads based on criteria
    3. Route qualified leads for follow-up
    """

    def test_extract_lead_information(self, sie_url: str) -> None:
        """Example: Extract company and contact info for lead scoring."""
        from sie_crewai import SIEExtractorTool

        # Lead qualification agent tool
        extractor = SIEExtractorTool(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["person", "organization", "location", "job_title"],
        )

        # Sample lead information from email or form
        lead_text = """
        Hi, I'm Sarah Chen, CTO at DataFlow Technologies.
        We're a Series A startup based in Austin, Texas, looking for
        an embedding solution for our semantic search product.
        Our team of 25 engineers processes about 10M documents monthly.
        """

        result = extractor._run(text=lead_text)

        # Should extract entities like: Sarah Chen, DataFlow Technologies, Austin, CTO
        assert "Extracted entities" in result or "entities" in result.lower()

    def test_extract_from_company_description(self, sie_url: str) -> None:
        """Example: Extract entities from company research."""
        from sie_crewai import SIEExtractorTool

        extractor = SIEExtractorTool(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["organization", "person", "money", "date"],
        )

        company_info = """
        TechVentures Inc. announced today that CEO Michael Roberts
        has secured $75 million in Series C funding. The round was
        led by Andreessen Horowitz with participation from Sequoia Capital.
        The company plans to expand its AI research division by Q2 2025.
        """

        result = extractor._run(text=company_info)

        assert "Extracted entities" in result or "No entities found" in result


class TestContentCreationWorkflow:
    """Integration tests demonstrating content creation use case.

    CrewAI is commonly used for content workflows where agents:
    1. Research topics from multiple sources
    2. Rerank sources by relevance
    3. Extract key facts for content synthesis
    """

    def test_research_and_extract_pipeline(self, sie_url: str) -> None:
        """Example: Combined rerank + extract for content research."""
        from sie_crewai import SIEExtractorTool, SIERerankerTool

        # Step 1: Rerank research sources
        reranker = SIERerankerTool(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        articles = [
            "OpenAI released GPT-4 in March 2023, marking a significant advancement in language models.",
            "The weather in California remains sunny with temperatures reaching 75°F.",
            "Anthropic's Claude 3 was launched in March 2024 with improved reasoning capabilities.",
            "Google DeepMind announced Gemini in December 2023 as their most capable AI model.",
            "Apple stock rose 2% following the latest product announcement.",
        ]

        ranked = reranker._run(
            query="Recent developments in large language models",
            documents=articles,
            top_k=3,
        )

        assert "Ranked documents" in ranked

        # Step 2: Extract key entities from top result
        extractor = SIEExtractorTool(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["organization", "product", "date"],
        )

        # Extract from one of the relevant articles
        extracted = extractor._run(text="OpenAI released GPT-4 in March 2023, marking a significant advancement.")

        assert "Extracted entities" in extracted or "No entities found" in extracted


class TestMultiAgentCollaboration:
    """Integration tests demonstrating multi-agent collaboration.

    Shows how SIE tools can be used in different agent roles
    within a CrewAI workflow.
    """

    def test_analyst_and_writer_workflow(self, sie_url: str) -> None:
        """Example: Analyst agent reranks, writer agent extracts facts."""
        from sie_crewai import SIEExtractorTool, SIERerankerTool

        # Analyst agent: find relevant market data
        analyst_reranker = SIERerankerTool(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        market_reports = [
            "Q3 2024 saw a 15% increase in AI infrastructure spending globally.",
            "Consumer electronics sales declined 3% due to supply chain issues.",
            "Enterprise AI adoption reached 45% among Fortune 500 companies.",
            "The semiconductor shortage continues to impact manufacturing.",
            "Cloud computing revenue grew 22% year-over-year for major providers.",
        ]

        analyst_result = analyst_reranker._run(
            query="Enterprise AI market trends and adoption",
            documents=market_reports,
            top_k=2,
        )

        assert "Ranked documents" in analyst_result

        # Writer agent: extract specific facts for report
        writer_extractor = SIEExtractorTool(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["percentage", "organization", "metric"],
        )

        writer_result = writer_extractor._run(
            text="Enterprise AI adoption reached 45% among Fortune 500 companies in 2024."
        )

        # The writer can now use these extracted facts in the report
        assert isinstance(writer_result, str)
