import os

from .config import Settings


class LLMClient:
    """Provider-agnostic async chat client. Claude via Anthropic; open models via Groq."""

    def __init__(self, s: Settings):
        self.provider = s.llm_provider
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=3)
            self.model = s.anthropic_model
        elif self.provider == "groq":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                max_retries=3,
            )
            self.model = s.groq_model
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    async def complete(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, dict]:
        if self.provider == "anthropic":
            r = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in r.content if b.type == "text")
            return text, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
        r = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        u = r.usage
        return r.choices[0].message.content or "", {
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
        }
