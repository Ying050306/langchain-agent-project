import os
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """You are a precise data assistant.
Use the 'read_pdf_text' tool to extract information from the provided file. 
Stay strictly within the facts found in the document."""


@tool
def read_pdf_text(file_name: str) -> str:
    """Reads all text from a local PDF file.
    """
    try:
        # Construct the path to the file in project folder
        reader = PdfReader(file_name)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        # Return the first 25k characters to stay within model limits
        return text[:25000] if text else "The PDF appears to be empty or an image-only scan."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google-genai",
    temperature=0.1,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[read_pdf_text],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

pdf_name = "ecommerce.pdf"

content = f"""Please read the file '{pdf_name}' and answer:
1) What are the main advantages and disadvantages listed?
2) What specific types of business models are mentioned?"""

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "pdf_task"}}
)

print("\n--- PDF Analysis Result ---")
print(agent_result["messages"][-1].content)