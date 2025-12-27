from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from abc import ABC
from typing import Any

from config import settings


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        output_type: type[Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.output_type = output_type

        self.agent = self._setup_agent()

    def _setup_agent(self):
        model = OpenRouterModel(
            self.model, provider=OpenRouterProvider(api_key=settings.openrouter_api_key)
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "retries": 2,
            "system_prompt": self.system_prompt,
        }
        if self.output_type:
            kwargs["output_type"] = self.output_type

        agent = Agent(**kwargs)

        return agent

    async def run(self, prompt: str, message_history: list | None = None):
        return await self.agent.run(prompt, message_history=message_history)


class SubAgent(BaseAgent):
    def register_as_tool(self, parent: Agent):
        tool_name = f"ask_{self.name.lower().replace(' ', '_')}"

        async def call_subagent(prompt: str):
            print(f"[{parent.name or 'Parent'} -> {self.name}] Delegating a task")
            response = await self.agent.run(prompt)
            return response.output

        call_subagent.__name__ = tool_name
        call_subagent.__doc__ = (
            f"Delegate task to {self.name}: {self.system_prompt or 'no description'}"
        )

        parent.tool_plain(call_subagent)
        return call_subagent
