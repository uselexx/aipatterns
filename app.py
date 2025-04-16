from typing import Dict, Optional, Union
import requests
from mcp import ClientSession
import chainlit as cl
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
)

# Tool registry
registered_tools = {
    "calculator": "http://localhost:5000/calculate",
    "summarizer": "http://localhost:5000/summarize",
}

# Custom assistant agent for tool use
class ToolAgent(AssistantAgent):
    def handle_tool(self, tool: str, input_text: str):
        endpoint = registered_tools.get(tool)
        if not endpoint:
            return f"❌ Tool '{tool}' not found."

        try:
            response = requests.post(endpoint, json={"input": input_text})
            if response.status_code == 200:
                return response.json().get("result", "⚠️ No result returned.")
            else:
                return f"❌ Tool error: {response.text}"
        except Exception as e:
            return f"❌ Tool call failed: {str(e)}"

# Simple RAG agent (mocked)
class RAGAgent(AssistantAgent):
    def handle_query(self, query: str):
        return f"📚 RAG response for: {query}"

# Router logic
def should_use_tool(input_text: str) -> Optional[str]:
    for tool in registered_tools:
        if tool in input_text.lower():
            return tool
    return None

# Chainlit-enabled assistant agent
class ChainlitAssistantAgent(AssistantAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f"*{self.name} ➡️ {recipient.name}:*\n\n{message}",
                author=self.name,
            ).send()
        )
        return super().send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

# Chainlit-enabled user agent with routing logic
class RouterAgent(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        reply = cl.run_sync(
            cl.AskUserMessage(content=prompt, timeout=60).send()
        )
        return reply["content"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        cl.run_sync(
            cl.Message(
                content=f"*{self.name} ➡️ {recipient.name}:*\n\n{message}",
                author=self.name,
            ).send()
        )
        return super().send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

    def route_message(self, user_input: str, tool_agent: ToolAgent, rag_agent: RAGAgent):
        tool_name = should_use_tool(user_input)
        if tool_name:
            return tool_agent, {"tool": tool_name, "input": user_input}
        else:
            return rag_agent, user_input

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    # Your connection initialization code here
    # This handler is required for MCP to work
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional

# Chainlit startup
@cl.on_chat_start
async def on_chat_start():
    # Ollama + Mistral config
    config_list = [
        {
            "model": "mistral",
            "base_url": "http://localhost:11434",
            "api_key": "ollama",
        }
    ]

    # Shared config: disable Docker
    exec_config = {
        "work_dir": "workspace",
        "use_docker": False,
    }

    tool_agent = ToolAgent(
        name="ToolAgent",
        llm_config={"config_list": config_list},
        code_execution_config=exec_config,
    )

    rag_agent = RAGAgent(
        name="RAGAgent",
        llm_config={"config_list": config_list},
        code_execution_config=exec_config,
    )

    router = RouterAgent(
        name="RouterAgent",
        code_execution_config=exec_config,
    )

    await cl.Message(content="✅ Agents initialized! Ask me something.").send()

    @cl.on_message
    async def handle_message(message: cl.Message):
        user_input = message.content
        target_agent, payload = router.route_message(user_input, tool_agent, rag_agent)

        if isinstance(payload, dict) and "tool" in payload:
            result = target_agent.handle_tool(payload["tool"], payload["input"])
        else:
            result = target_agent.handle_query(payload)

        await cl.Message(content=f"✅ Result:\n\n{result}").send()
