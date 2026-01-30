"""LLM service supporting both Anthropic Claude and OpenAI."""

import logging
from typing import Literal, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """Service for LLM-powered analysis using Claude or OpenAI."""

    def __init__(
        self,
        provider: Optional[Literal["anthropic", "openai"]] = None,
    ):
        """
        Initialize LLM service.

        Args:
            provider: LLM provider to use (anthropic or openai)
                      Defaults to settings.llm_provider
        """
        self.provider = provider or settings.llm_provider
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        """Get Anthropic client (lazy loaded)."""
        if self._anthropic_client is None:
            from anthropic import Anthropic

            self._anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
        return self._anthropic_client

    def _get_openai_client(self):
        """Get OpenAI client (lazy loaded)."""
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using the configured LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if self.provider == "anthropic":
            return await self._generate_anthropic(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            return await self._generate_openai(
                prompt, system_prompt, max_tokens, temperature
            )

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Anthropic Claude."""
        import asyncio

        client = self._get_anthropic_client()

        # Run in thread pool (sync client)
        loop = asyncio.get_event_loop()

        def _call():
            kwargs = {
                "model": settings.claude_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
            return response.content[0].text

        return await loop.run_in_executor(None, _call)

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using OpenAI."""
        import asyncio

        client = self._get_openai_client()

        loop = asyncio.get_event_loop()

        def _call():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        return await loop.run_in_executor(None, _call)

    async def analyze_video_content(
        self,
        transcript: str,
        analysis_type: Literal["summary", "topics", "sentiment", "suggestions"] = "summary",
    ) -> dict:
        """
        Perform video content analysis.

        Args:
            transcript: Video transcript
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results as dictionary
        """
        prompts = {
            "summary": f"""Summarize this video transcript in 2-3 sentences.
Focus on the main topic and key takeaways.

Transcript:
{transcript[:3000]}

Return only the summary.""",

            "topics": f"""Extract the main topics discussed in this video.
List 3-5 topics with brief descriptions.

Transcript:
{transcript[:3000]}

Return as JSON: {{"topics": [{{"name": "topic", "description": "brief description"}}]}}""",

            "sentiment": f"""Analyze the overall sentiment and tone of this video.

Transcript:
{transcript[:3000]}

Return as JSON: {{"sentiment": "positive/negative/neutral", "tone": "educational/entertaining/informative/persuasive", "confidence": 0.0-1.0}}""",

            "suggestions": f"""Based on this video content, suggest improvements for engagement.

Transcript:
{transcript[:3000]}

Provide 3 specific suggestions to make this content more engaging.
Return as JSON: {{"suggestions": ["suggestion1", "suggestion2", "suggestion3"]}}""",
        }

        prompt = prompts.get(analysis_type, prompts["summary"])
        response = await self.generate(prompt, max_tokens=500)

        # Try to parse JSON response
        import json
        import re

        try:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Return raw response if not JSON
        return {"result": response}

    async def generate_clip_description(
        self,
        transcript_excerpt: str,
        virality_scores: dict,
    ) -> str:
        """Generate engaging description for a clip."""
        prompt = f"""Write a compelling 1-2 sentence description for this video clip.
Make it engaging and highlight what makes this clip interesting.

Clip transcript:
"{transcript_excerpt[:500]}"

Key strengths:
- Emotional impact: {virality_scores.get('emotional_resonance', 0):.0f}/20
- Shareability: {virality_scores.get('shareability', 0):.0f}/20
- Uniqueness: {virality_scores.get('uniqueness', 0):.0f}/20

Return only the description."""

        return await self.generate(prompt, max_tokens=100)

    async def improve_hook(
        self,
        original_hook: str,
        context: str,
    ) -> list[str]:
        """Generate improved hook alternatives."""
        prompt = f"""The current video opening is:
"{original_hook}"

Context from the video:
{context[:500]}

Generate 3 more engaging alternative openings that:
1. Create curiosity
2. Are concise (under 15 words)
3. Directly address the viewer

Return only the 3 alternatives, one per line."""

        response = await self.generate(prompt, max_tokens=150)

        # Parse response
        alternatives = [
            line.strip().strip('"').strip("'").lstrip("123.-) ")
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        return alternatives[:3]


# Singleton instances per provider
_llm_services = {}


def get_llm_service(provider: Optional[str] = None) -> LLMService:
    """Get LLM service instance."""
    provider = provider or settings.llm_provider
    if provider not in _llm_services:
        _llm_services[provider] = LLMService(provider=provider)
    return _llm_services[provider]
