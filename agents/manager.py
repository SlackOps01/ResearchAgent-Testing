from .specialized_agents import CriticAgent, Researcher, Orchestrator, Writer
from pydantic_ai import AgentRunResult


class AgentManager:
    def __init__(self) -> None:
        self.researcher = Researcher()
        self.orchestrator = Orchestrator() 
        self.writer = Writer()
        self.critique = CriticAgent()
        self._setup_orchestration()
        
    def _setup_orchestration(self):
        self.researcher.register_as_tool(self.orchestrator.agent)
        self.writer.register_as_tool(self.orchestrator.agent)
        self.critique.register_as_tool(self.orchestrator.agent)
        
    async def handle_request(self, prompt: str, message_history: list | None = None) -> AgentRunResult:
        return await self.orchestrator.run(prompt, message_history)