import ast

from typing import List, Sequence, Any

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.ui import Console

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

def print_items(name: str, result: Any) -> None:
    """Print items with formatting.

    Args:
        name: Category name (tools/resources/prompts)
        result: Result object containing items list
    """
    print("", f"Available {name}:", sep="\n")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")

class WeatherAgent(BaseChatAgent):
    """Agent that can answer questions about the weather."""

    def __init__(
        self,
        name: str,
        description: str = "An assistant that can provide information about the weather.",
    ):
        super().__init__(name=name, description=description)
        self._message_history: List[BaseChatMessage] = []
        # self.model = OllamaLLM(model="mistral", base_url="http://localhost:11434")
        # self._message_history: List[BaseChatMessage] = []

    
    
    async def async_init(self) -> None:
        server_params = SseServerParams(
            url="http://localhost:5000",
            timeout=30,  # Connection timeout in seconds
        )

        # Create an agent that can use the weather API
        model_client = OllamaChatCompletionClient(
            model="mistral",
            host = 'http://localhost:11434',
        )

        tools = await mcp_server_tools(server_params)

        self.agent = AssistantAgent(
            name="weatherman",
            model_client=model_client,
            tools=tools,
            system_message="You are a helpful weather assistant.",
        )

    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        self._message_history.extend(messages)
        assert isinstance(self._message_history[-1], TextMessage)
        last_message = str(self._message_history[-1].content)

        print("Last message:", last_message)
        response = await get_weather_summary(self.agent, last_message, cancellation_token)
        # # Create a new message with the result.
        response_message = TextMessage(content=response, source=self.name)

        # # Return the response.
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        return

async def get_weather_summary(agent: AssistantAgent, message: str, cancellation_token: CancellationToken) -> str:
    """Get the weather summary from the agent."""
    stream = agent.run_stream(task=message, cancellation_token=cancellation_token)

    final_result = None
    async for m in stream:
        if isinstance(m, TaskResult):
            final_result = m

    if final_result:
        # Extract the last summary message from the result
        for msg in reversed(final_result.messages):
            if getattr(msg, "type", "") == "ToolCallSummaryMessage":
                print("Weather summary:", msg.content)
                return f"Weather summary: {msg.content}"
                # break

    # result = None

    # async for event in agent.run_stream(
    #     task=message,
    #     cancellation_token=cancellation_token
    # ):
    #     # Optional: print intermediate messages or events
    #     print("Received:", event)

    #     # TaskResult is the final item in the stream
    #     if isinstance(event, TaskResult):
    #         result = event

    # if result:
    #     final_messages = result.messages
    #     if final_messages:
    #         # Return the content of the last message
    #         return final_messages[-1]
    #     else:
    #         print("No messages in result.")
    #         return None
    # else:
    #     print("No TaskResult received.")
    #     return None
