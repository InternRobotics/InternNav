from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg


@Agent.register('seq2seq')
class Seq2SeqAgent(Agent.agents['cma']):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
