# from langchain_ollama import OllamaLLM  # Using Mistral, replace with the actual setup
# from langchain_community.tools import QuerySQLDataBaseTool  # Tool for SQL Queries
# from langchain_community.vectorstores import FAISS  # Tool for FAISS vector search
# from langchain_community.document_loaders import TextLoader  # Document loader tool
# import os
# import requests
# from dotenv import load_dotenv


# # Initialize OpenAI API key (either from env or direct)
# openai_api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key
# # Initialize the SLM model (Mistral or another SLM)
# llm = OllamaLLM(
#     base_url="http://localhost:11434",  # Ollama's local API endpoint or the actual Mistral endpoint
#     model="mistral"  # Specify the model name here
# )

# # --- TOOL SETUP ---
# # Define the available tools with descriptions
# tools = {
#     "faq_tool": {
#         "description": "This tool can answer frequently asked questions about the campus system, such as registration details, working hours, etc.",
#         "tool": "faq_tool"
#     },
#     "sql_tool": {
#         "description": "This tool can execute SQL queries against a campus database to retrieve factual information like sales data, product counts, etc.",
#         "tool": "sql_tool"
#     },
#     "blog_tool": {
#         "description": "This tool fetches recent blog posts and campus announcements from the campus blog API.",
#         "tool": "blog_tool"
#     }
# }

# # --- AGENT THAT USES SLM MISTRAL FOR DECISION MAKING ---
# class DynamicAgent:
#     def __init__(self, llm, tools):
#         self.llm = llm
#         self.tools = tools

#     def get_tool_decision(self, query: str):
#         prompt = f"""
#                 You are a helpful AI assistant responsible for selecting the most appropriate tool to handle a user query.

#                 Your task is to **choose one tool name only**, based on the user's question and the descriptions of the available tools. 
#                 Do not explain your choice. Only respond with the exact tool name — nothing else.

#                 User Query:
#                 {query}

#                 Available Tools:
#                 """ + "\n".join([f"- {name}: {info['description']}" for name, info in self.tools.items()]) + "\n\n" + \
#                 "Reply with just the tool name (e.g., `faq_tool`, `sql_tool`, `blog_tool`)."
#         for tool_name, tool_info in self.tools.items():
#             prompt += f"- {tool_name}: {tool_info['description']}\n"
        
#         prompt += "\nWhich tool should be used to handle this query?"

#         try:
#             response = self.llm.generate([prompt])  # Generate response
#             selected_tool = response.generations[0][0].text.strip()  # Extract and strip the generated text
#             print(f"Mistral decision: {selected_tool}")
#             return selected_tool
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return None

#     def run_tool(self, selected_tool: str, query: str):
#         """Runs the tool chosen by the model."""
#         if selected_tool in self.tools:
#             tool_info = self.tools[selected_tool]
#             # Here we would invoke the selected tool with the query
#             # For demonstration, we'll mock the behavior of tools

#             if selected_tool == "faq_tool":
#                 # Mocking FAQ tool (replace with actual functionality)
#                 return f"FAQ Tool Response: Here's the answer to your query: '{query}' (simulated response)"
#             elif selected_tool == "sql_tool":
#                 # Mocking SQL tool (replace with actual functionality)
#                 return f"SQL Tool Response: Querying the database for '{query}' (simulated response)"
#             elif selected_tool == "blog_tool":
#                 # Mocking Blog tool (replace with actual functionality)
#                 return f"Blog Tool Response: Fetching blog posts related to '{query}' (simulated response)"
#         else:
#             return "Sorry, I couldn't find a tool that can handle this query."

#     def handle_query(self, query: str):
#         selected_tool = self.get_tool_decision(query)
#         response = self.run_tool(selected_tool, query)
#         return response

# # Initialize the agent
# agent = DynamicAgent(llm=llm, tools=tools)

# # --- ENTRYPOINT ---
# def run_chat(query):
#     print(f"User: {query}")
#     response = agent.handle_query(query)
#     print(f"Agent: {response}")

# # Example of running the agent
# if __name__ == "__main__":
#     user_query = input("Enter your query: ")
#     run_chat(user_query)


#### CHAINLIT IMPLEMENTATION ####
from typing import List
import chainlit as cl
# from langchain_community.llms import Ollama

from autogen_ext.models.ollama import OllamaChatCompletionClient

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.ui import Console

from Arithmetic_agent import ArithmeticAgent
from rag_agent import RAGAgent


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="🤖 Initializing agents with smart routing...").send()
    add_agent = ArithmeticAgent("add_agent", "Adds 1 to the number.", lambda x: x + 1)
    multiply_agent = ArithmeticAgent("multiply_agent", "Multiplies the number by 2.", lambda x: x * 2)
    subtract_agent = ArithmeticAgent("subtract_agent", "Subtracts 1 from the number.", lambda x: x - 1)
    divide_agent = ArithmeticAgent("divide_agent", "Divides the number by 2 and rounds down.", lambda x: x // 2)
    identity_agent = ArithmeticAgent("identity_agent", "Returns the number as is.", lambda x: x)

    rag_agent = RAGAgent("rag_agent")

    termination_condition = MaxMessageTermination(10)

    client = OllamaChatCompletionClient(
        model="mistral",
        host = 'http://localhost:11434',
    )


    selector_prompt = """Select an agent to perform task.

        {roles}

        Current conversation context:
        {history}

        Read the above conversation, then select an agent from {participants} to perform the next task.
        Make sure the planner agent has assigned tasks before other agents start working.
        Only select one agent.
        """
    
    selector_group_chat = SelectorGroupChat(
        [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent, rag_agent],
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
        TextMessage(content=msg.content, source="user"),
        # TextMessage(content="10", source="user"),
    ]
    # task: List[BaseChatMessage] = [
    #     TextMessage(content="Apply the operations to turn the given number into 25.", source="user"),
    #     TextMessage(content="10", source="user"),
    # ]
    selector_group_chat = cl.user_session.get("selector_group_chat")
    if not selector_group_chat:
        await cl.Message(content="⚠️ Selector group chat not initialized.").send()
        return
    
    # Get the selected tool and its response
    stream = selector_group_chat.run_stream(task=task)

    # await Console(stream)
    latest_message = None
    print("Waiting for messages...")
    async for message in stream:
        latest_message = message

    if latest_message and latest_message.messages:
        final_text = latest_message.messages[-1].content
        await cl.Message(content=f"🧠 {final_text}").send()
