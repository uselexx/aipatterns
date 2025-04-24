#### CHAINLIT IMPLEMENTATION ####
# Chainlit is used to create a web-based interface for the agents.
# It allows users to interact with the agents through a chat interface.
# This implementation demonstrates how to use agents as a design pattern.
# The agents are implemented as classes that inherit from the abstract BaseAgent class.
# The RAG agent uses RAG to retrieve relevant information from a knowledge base.
# The Arithmetic agents perform basic arithmetic operations.

from typing import List
import chainlit as cl

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient

from Arithmetic_agent import ArithmeticAgent
from weather_agent import WeatherAgent
from rag_agent import RAGAgent


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="🤖 Initializing agents with smart routing...").send()

    weather_agent = WeatherAgent("weather_agent")
    await weather_agent.async_init()

    # Initialize the agents with their respective functions and descriptions.
    add_agent = ArithmeticAgent("add_agent", "Adds 1 to the number.", lambda x: x + 1)
    multiply_agent = ArithmeticAgent("multiply_agent", "Multiplies the number by 2.", lambda x: x * 2)
    subtract_agent = ArithmeticAgent("subtract_agent", "Subtracts 1 from the number.", lambda x: x - 1)
    divide_agent = ArithmeticAgent("divide_agent", "Divides the number by 2 and rounds down.", lambda x: x // 2)
    identity_agent = ArithmeticAgent("identity_agent", "Returns the number as is.", lambda x: x)
    # Initialize the RAG agent
    rag_agent = RAGAgent("rag_agent")

    termination_condition = MaxMessageTermination(2)
    # Initialize the client for the Mistral model using Ollama
    # Ensure you have the Ollama server running locally or replace with the actual endpoint.
    client = OllamaChatCompletionClient(
        model="mistral",
        host = 'http://localhost:11434',
    )

    # Prompt for the selector agent to choose which agent to use.
    # This prompt will be used to guide the selection process.
    selector_prompt = """Select an agent to perform task.

        {roles}

        Current conversation context:
        {history}

        Read the above conversation, then select an agent from {participants} to perform the next task.
        Make sure the planner agent has assigned tasks before other agents start working.
        Only select one agent.
        """
    
    # The selector group chat will manage the conversation and agent selection.
    selector_group_chat = SelectorGroupChat(
        [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent, rag_agent, weather_agent],
        model_client=client,
        termination_condition=termination_condition,
        allow_repeated_speaker=True,  # Allow the same agent to speak multiple times, necessary for this task.
        selector_prompt=selector_prompt,
    )

    cl.user_session.set("selector_group_chat", selector_group_chat)

    await cl.Message(content="🤖 Agents initialized. You can start asking questions!").send()

@cl.on_message
async def on_message(msg: cl.Message):

    # Run the selector group chat with a given task and stream the response.
    print(f"User: {msg.content}")
    task: List[BaseChatMessage] = [
        TextMessage(content=msg.content, source="user")
    ]
    # task: List[BaseChatMessage] = [
    #     TextMessage(content="Apply the operations to turn the given number into 25.", source="user"),
    #     TextMessage(content="10", source="user"),
    # ]
    selector_group_chat = cl.user_session.get("selector_group_chat")
    if not selector_group_chat:
        await cl.Message(content="⚠️ Selector group chat not initialized.").send()
        return
    
    # Get the stream of messages from the selector group chat.
    # This will allow us to receive messages in real-time as they are generated.
    stream = selector_group_chat.run_stream(task=task)

    # await Console(stream)
    latest_message = None
    print("Waiting for messages...")
    async for message in stream:
        latest_message = message

    # Return the final result to the user.
    if latest_message and latest_message.messages:
        final_text = latest_message.messages[-1].content
        await cl.Message(content=f"🧠 {final_text}").send()
