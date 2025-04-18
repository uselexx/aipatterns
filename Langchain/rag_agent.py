import os
import shutil

from typing import List, Sequence

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings


from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

CHROMA_PATH = "chroma"
PDF_PATH = "data/pdf"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class RAGAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        description: str = "A helpful AI assistant that can answer questions with the help of a PDF document.",
        chroma_path=CHROMA_PATH,
        pdf_path=PDF_PATH
    ):
        super().__init__(name=name, description=description)
        self.chroma_path = chroma_path
        self.pdf_path = pdf_path
        self.model = OllamaLLM(model="mistral", base_url="http://localhost:11434")
        self._message_history: List[BaseChatMessage] = []

        # Setup RAG
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        self._message_history.extend(messages)
        # Parse the number in the last message.
        assert isinstance(self._message_history[-1], TextMessage)
        last_message = str(self._message_history[-1].content)

        # Apply the operator function to the number.
        prompt = self.generate_prompt(last_message)
        print(f"Prompt: {prompt}")
        response = self.model.invoke(prompt)
        # Create a new message with the result.
        response_message = TextMessage(content=response, source=self.name)

        # Return the response.
        return Response(chat_message=response_message)

 

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        await self._message_history.clear()


    def generate_prompt(self, query: str) -> str:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_score(query, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        return prompt
    
    # def handle_query(self, query: str):
    #     embedding_function = get_embedding_function()
    #     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #     # Search the DB.
    #     results = db.similarity_search_with_score(query, k=5)

    #     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    #     prompt = prompt_template.format(context=context_text, question=query)

    #     self.update_system_message(prompt)
        

    #     # here I want to use the llm model to generate the response
    #     response_text = self.model.invoke(prompt)

    #     sources = [doc.metadata.get("id", None) for doc, _score in results]
    #     formatted_response = f"Response: {response_text}\nSources: {sources}"
    #     print(formatted_response)
    #     return response_text


def load_documents():
    document_loader = PyPDFDirectoryLoader(PDF_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings