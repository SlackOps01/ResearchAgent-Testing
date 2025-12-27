from pydantic_ai import Agent
from .base import SubAgent
from pydantic import BaseModel, Field
import json


class Orchestrator(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Orchestrator",
            model="mistralai/devstral-2512:free",
            system_prompt="You are an orchestrator agent, you are never to solve any task on your own rather delegate to relevant subagents, and always make sure responses go to the writing agent for appropriate formatting and respond with the output of the writer agent and pass this output to critique agent to see if a rewrite is needed and only respond with a final output when the critic is satisfied always use the critique agent to validate output",
        )


class Researcher(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Researcher agent",
            model="mistralai/ministral-8b-2512",
            system_prompt="You are researcher agent build to research topics and make research on any given prompt",
        )


class Writer(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Writer",
            model="mistralai/ministral-8b-2512",
            system_prompt="You are writing agent built to write formally based on a given prompt",
        )


class CritiqueFormat(BaseModel):
    score: int = Field(
        ..., description="A score from 1-10 on how satisfied you are with the output"
    )
    cons: str = Field(..., description="Bad parts of the essay that needs improvements")
    pros: str = Field(..., description="The good parts of the essay")
    summary: str = Field(..., description="A summary of your thoughts on the essay")
    satisfied: bool = Field(..., description="True if score is greater than 7")


class CriticAgent(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Critic agent",
            model="mistralai/ministral-8b-2512",
            system_prompt="You are a critic agent whose job is critique a piece of writing",
            output_type=CritiqueFormat,
        )

    def register_as_tool(self, parent: Agent):
        tool_name = f"ask_{self.name.lower().replace(' ', '_')}"

        async def call_subagent(prompt: str):
            print(f"[{parent.name or 'Parent'} -> {self.name}] Delegating a task")
            response = await self.agent.run(prompt)
            response_output: CritiqueFormat = response.output
            print(f"{json.dumps(response_output.model_dump(), indent=4)}")
            return response.output

        call_subagent.__name__ = tool_name
        call_subagent.__doc__ = f"Delegate to {self.name}: {self.system_prompt}"

        parent.tool_plain(call_subagent)
        return call_subagent
