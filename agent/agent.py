from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
import arxiv
from typing import List, Dict,AsyncGenerator
from autogen_agentchat.teams import RoundRobinGroupChat 
from autogen_core.models import ModelInfo
import asyncio

load_dotenv()


def arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for academic papers on arXiv based on the given query.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing paper details.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results:List[Dict]=[]
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": ", ".join(author.name for author in result.authors),
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        })
    
    return results

openai_router_brain = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
    api_key=os.getenv("GEMINI_API_KEY")
)

research_agent = AssistantAgent(
    name="ResearchAgent",
    description="An agent that searches for academic papers on arXiv.",
    model_client=openai_router_brain,
    tools=[arxiv_search],
    system_message=(
        "You are an expert researcher. When you receive a topic, search for academic papers on arXiv "
        "and return a JSON list of papers with the following fields:\n"
        "- title\n"
        "- authors\n"
        "- year\n"
        "- summary (abstract)\n"
        "Return only the JSON list, no other text."
    )
)

summarizer_agent = AssistantAgent(
    name="SummarizerAgent",
    description="An agent that summarizes text.",
    model_client=openai_router_brain,
    system_message=(
        "Write a literature review in Markdown for these papers. "
        "Format the report as follows:\n\n"
        "## **Introduction**\n"
        "**Write a 2-3 sentence introduction to the topic. Make this section bold and clear.**\n\n"
        "## Papers\n"
        "For each paper below, use a circle bullet (●) and format as follows, with each paper in a separate paragraph. Do NOT return a JSON list, only return the Markdown report:\n"
        "● 👉 [**[{title}]({pdf_url})**]  "
        "\n   **Authors**: {authors}  "
        "\n   **Year**: {published}  "
        "\n   **Summary**: {summary}  "
        "\n\n"
        "Add extra spacing between papers for readability. Make the bullets visually appealing and easy to scan. Use bold for titles, monospace for URLs, and italics for authors/year/summary labels.\n\n"
        "## **Conclusion**\n"
        "**End with a brief, insightful conclusion in bold.**\n\n"
    )
)


teams = RoundRobinGroupChat(
    participants=[research_agent, summarizer_agent],
    max_turns=2
)


async def run_team():
    task = 'Conduct a literature review on the topic - Autogen and return exactly 2 papers.'
    last_content = None
    async for msg in teams.run_stream(task=task):
        # Only print the content from the SummarizerAgent
        if hasattr(msg, 'source') and getattr(msg, 'source', None) == 'SummarizerAgent' and hasattr(msg, 'content'):
            last_content = msg.content
    if last_content:
        print(f"content={last_content}")
    else:
        print("No summarizer agent content found.")
        
if __name__ == "__main__":
    # Use the agent team workflow (researcher and summarizer agents)
    asyncio.run(run_team())



