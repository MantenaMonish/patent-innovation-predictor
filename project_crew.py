import os
from datetime import datetime, timedelta

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from crewai import LLM

import requests
from chromadb_client import create_chromadb_client
from embedding import get_embedding

# Ollama Helpers

def check_ollama_availability():
    '''
    Check if Ollama is running and return all the available models
    '''
    try:
        response = requests.get("http://localhost:11434/api/tags",timeout=5)
        response.raise_for_status()
        models = response.json().get("models",[])
        return [ model.get("name") for model in models if model.get("name")]
    except Exception as e:
        print(f"Error Connecting to Ollama: {e}")
        return []
    
def test_model(model_name):
    '''
    To test if the model responds by directly hitting Ollama API, since CrewAI's LLM uses OpenAI by default while testing(U need OpenAI's Key).
    '''
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Say Hello",
                "stream": False
            },
            timeout=120
        )
        if response.status_code == 200:
            return bool(response.json().get("response"))
        return False
    except Exception as e:
        print(f"Error while testing the model {model_name} : {e}")
        return False
    
# Tools

class search_patent_tool(BaseTool):
    name: str = "search_patents"
    description: str = "Search for patents matching a query using semantic vector search"

    def _run(self, query: str, top_k: int=20) -> str:
        try:
            client = create_chromadb_client("localhost", 8000)
            collection = client.get_collection(name="patents")

            # Semantic search — ChromaDB embeds the query_text automatically
            query_embedding=get_embedding(query)

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents","metadatas","distances"]
            )

            return format_results_tool(response)
        
        except Exception as e:
            print(f"Error Searching Patents: {str(e)}")


class SearchPatent_ByDateRange_Tool(BaseTool):
    name: str = "search_patents_by_date_range"
    description: str = "Search for patents in a specific date range"

    def _run(self, query: str, start_date: str, end_date: str, top_k: int = 30) -> str:
        '''
        Chromadb supports metadata filtering via 'where' clause.
        We filter publication_date using $gte and $lte operators - 
        equivalent to Opensearch range filter

        Args:
            query: Search query text
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            top_k: Number of results to return
        '''
        try:
            client = create_chromadb_client("localhost",8000)
            collection = client.get_collection(name="patents")

            query_embedding = get_embedding(query)
            
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={
                    "$and": [
                        {"publication_date":{"$gte":start_date}},
                        {"publication_date":{"$lte":end_date}}
                    ]
                },
                include=["documents","metadatas","distances"]
            )

            return format_results_tool(response)

        except Exception as e:
            print(f"Error searching patents by date range: {str(e)}")
            
class analyzepatent_tool(BaseTool):
    name: str = "analyze_patent_trends"
    description: str = "Analyze trends in patent data"

    def _run(self, patents_data: str) -> str:
        return f"Analysis of Patent Trend : {patents_data}" 
    
# Helpers - To format the Chromadb response

def format_results_tool(response) -> str:
    '''
    Covert the Chromadb data stored in parallel lists into readable string for llm to understand. 
    '''
    ids = response.get("ids",[[]])[0]
    documents = response.get("documents",[[]])[0]
    metadatas = response.get("metadatas",[[]])[0]
    distances = response.get("distances",[[]])[0]

    if not ids:
        return "No Patents Found."
    
    formmatted_results = []
    for i,(doc,meta,dist) in enumerate(zip(documents,metadatas,distances)):
        formmatted_results.append(
            f"{i+1}. Title: {meta.get('title', 'N/A')}\n"
            f"   Date: {meta.get('publication_date', 'N/A')}\n"
            f"   Patent ID: {meta.get('patent_id', 'N/A')}\n"
            f"   Similarity: {round(1 - dist, 4)}\n"   # convert distance → similarity score (Need to do this while using Chromadb)
            f"   Abstract: {doc[:200]}...\n"
        )

    return '\n'.join(formmatted_results)
    
# Crew

def create_patent_analyze_crew(model_name):
    '''
    Create a CrewAI crew for patent analysis using Ollama

    Args:
        model_name(str): Name of the Ollama model to use

    Returns:
        Crew: A CrewAI crew configures for patent analysis
    '''
    available_models = check_ollama_availability()
    if not available_models:
        raise RuntimeError(
            "Ollama service is unavailable, Make sure Ollama is running."
        )
    
    if not test_model(model_name):
        raise RuntimeError(f"Model {model_name} is not responding to the test prompts.")
    
    print("Model found and tested successfully")

    llm = LLM(model=f"ollama/{model_name}", base_url="http://localhost:11434")

    tools = [
            search_patent_tool(),
            SearchPatent_ByDateRange_Tool(),
            analyzepatent_tool(),
    ]

    # Agents
    research_director = Agent(
        role ="Research Director",
        goal ="Coordinate research efforts and define the scope of patent analysis",
        backstory ="You are an experienced research director who specializes in technological innovation analysis.",
        verbose = True,
        allow_delegation = False,
        llm = llm,
        tools = tools,
    )
    
    patent_retriever = Agent(
        role ="Patent Retriever",
        goal ="Find and retrieve the most relevant patents related to the research area",
        backstory ="You are a specialized patent researcher with expertise in information retrieval systems.",
        verbose = True,
        allow_delegation = False,
        llm = llm,
        tools = tools,
    )

    data_analyst = Agent(
        role ="Patent Data Analyst",
        goal ="Analyze patent data to identify trends, patterns, and emerging technologies",
        backstory ="You are a data scientist specializing in patent analysis with years of experience in technology forecasting.",
        verbose = True,
        allow_delegation = False,
        llm = llm,
        tools = tools,
    )

    innovation_forecaster = Agent(
        role ="Innovation Forecaster",
        goal ="Predict future innovations and technologies based on patent trends",
        backstory ="You are an expert in technological forecasting with a track record of accurate predictions in emerging technologies.",
        verbose = True,
        allow_delegation = False,
        llm = llm,
        tools = tools,
    )

    # Tasks

    task1 = Task(
        description='''
        Define a research plan for lithium battery patents:
        1. Key technology areas to focus on
        2. Time periods for analysis (focus on last 3 years)
        3. Specific technological aspects to analyze
        ''',
        expected_output='''
        A research plan with focus areas, time periods, and key technological aspects.
        ''',
        agent=research_director,

    )

    task2 = Task(
        description='''
        Using the research plan, retrieve patents related to lithium battery technology from the last 3 years.
        Use the search_patents and search_patents_by_date_range tools to gather comprehensive data.
        Focus on the most relevant and innovative patents.
        Group patents by sub-technologies within lithium batteries.
        Provide a summary of the retrieved patents, including:
        - Total number of patents found
        - Key companies/assignees
        - Main technological categories
        ''',
        expected_output='''
        A comprehensive patent retrieval report containing:
        - Summary of total patents found
        - List of key patents grouped by sub-technology
        - Analysis of top companies/assignees
        - Overview of main technological categories
        - List of the most innovative patents with summaries
        ''',
        agent=patent_retriever,
        dependencies=[task1],
    )

    task3 = Task(
        description='''
        Analyze the retrieved patent data to identify trends and patterns:
        1. Identify growing vs. declining areas of innovation
        2. Analyze technology evolution over time
        3. Identify key companies and their focus areas
        4. Determine emerging sub-technologies within lithium batteries
        5. Analyze patent claims to understand technological improvements
        ''',
        expected_output='''
        A trend analysis report containing:
        - Identification of growing vs. declining technology areas
        - Timeline of technology evolution
        - Company focus analysis
        - Emerging sub-technologies list
        - Technical improvement trends
        - Data-backed conclusions on innovation patterns
        ''',
        agent=data_analyst,
        dependencies=[task2],
    )

    task4 = Task(
        description='''
        Based on the patent analysis, predict future innovations in lithium battery technology:
        1. Identify technologies likely to see breakthroughs in the next 2-3 years
        2. Recommend specific areas for R&D investment
        3. Predict which companies are positioned to lead innovation
        4. Identify potential disruptive technologies
        5. Outline specific technical improvements likely to emerge
        ''',
        expected_output='''
        A future innovation forecast containing:
        - Predicted breakthrough technologies for next 2-3 years
        - Prioritized list of R&D investment areas
        - Companies likely to lead future innovation
        - Potential disruptive technologies and their impact
        - Timeline of expected technical improvements
        - Justification for all predictions based on patent data
        ''',
        agent=innovation_forecaster,
        dependencies=[task3],
    )

    crew = Crew(
        agents=[research_director,patent_retriever,data_analyst,innovation_forecaster],
        tasks=[task1,task2,task3,task4],
        verbose=True,
        process=Process.sequential,
        cache=False,
        max_rpm=10,
    )

    return crew

# Entry point

def run_patent_analyzer(research_area, model_name):
    '''
    Run the patent analyzer crew for the specified research area.
    '''
    try:
        crew = create_patent_analyze_crew(model_name)
        result = crew.kickoff(inputs={"research_area":research_area})

        if hasattr(result,"output"):
            return result.output
        elif hasattr(result,"result"):
            return result.result
        else:
            return str(result)
    except Exception as e:
        return(
            f"Analysis failed: {str(e)}\n\nTroubleshooting tips:\n"
            + "1. Make sure Ollama is running: 'ollama serve'\n"
            + "2. Pull a compatible model: 'ollama pull llama3' or 'ollama pull mistral'\n"
            + "3. Make sure ChromaDB is running: 'docker-compose up -d'\n"
            + "4. Make sure patents are indexed: run ingestion.py first\n"
            + "5. Check ChromaDB connection at http://localhost:8000"
        )


if __name__ == '__main__':
    research_area = input("Enter the research area to analyze (default: Lithium Battery): ")
    if not research_area:
        research_area = "Lithium Battery"

    model_name = input("Enter the Ollama model to use (default: tinyllama): ")
    if not model_name:
        model_name = "tinyllama"

    result = run_patent_analyzer(research_area, model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"

    if not isinstance(result,str):
        result = str(result)

    with open(filename,"w") as f:
        f.write(result)

    print(f"Analysis completed and saved to {filename}")