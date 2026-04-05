import os
from datetime import datetime

import requests
from dotenv import load_dotenv

from chromadb_client import create_chromadb_client
from embedding import get_embedding
from search_tool import hybrid_search, semantic_search, keyword_search, iterative_search
from project_crew import run_patent_analyzer

def display_main_menu():
    ''' Display The Main Menu Options '''
    print("\n"+ "=" * 60)
    print("Patent Innovation Predictor")
    print("\n"+ "=" * 60)
    print("1. Run Complete patent trend analysis and forecasting")
    print("2. Search for specific patents")
    print("3. Iterative patent exploration")
    print("4. View System Status")
    print("5. Exit")
    print("-" * 60)
    return input("Select the following options: ")


def run_complete_analysis():
    ''' Run the complete patent trend analysis using crewai'''
    print("\nRunning Complete Patent Analysis")
    print("This may take few mins based on the data volume being analysed")

    research_name = input("Enter the research area u want to perform trend analysis on (Default: Lithium Battery): ")
    if not research_name:
        research_name= "Lithium Battery"

    model_name = input("Enter the model name that u want to use (Default: tinyllama):")
    if not model_name:
        model_name = "tinyllama"

    print(f"\nStarted Patent Trend Analysis on the topic: {research_name}")
    print(f"We are using the following model: {model_name}")
    print("Agents are now processing the data...\n")
    
    try:
        result = run_patent_analyzer(research_name,model_name)

        if not isinstance(result,str):
            result = str(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patent_analysis_{timestamp}.txt"
        with open(filename,"w") as f:
            f.write(result)

        print(f"\nAnalysis completed and saved to {filename}")

        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(result[:500] + "...\n")

    except Exception as e:
        print(f"Error while analysis: {e}")

def search_specific_patents():
    '''Search Specific Patents in Chromadb'''
    print("\n Specific Search")
    print("=" * 60)
    query = input("Enter the search query you want to search in ChromaDB: ")
    if not query:
        print("We cannot proceed without a valid search query")
        return
    
    search_type = input("Enter the search type (1. Keyword Search , 2. Semantic Search , 3. Hybrid Search [Default] ):")
    if not search_type:
        search_type = "3"
    
    try:
        results = []
        if search_type == "1":
            results = keyword_search(query)
        elif search_type == "2":
            results = semantic_search(query)
        elif search_type == "3":
            results = hybrid_search(query)

        # To Display Results
        # ChromaDB provides data in folowing dict format: metadatas, documents, distances
        for i,result in enumerate(results):
            meta = result["metadata"]
            print(f"{i+1}. {meta.get('title','N/A')}")
            print(f" Similarity Score: {round(1-result['distance'],4)}")
            print(f" Date: {meta.get('publication_date','N/A')}")
            print(f" Patent_ID: {meta.get('patent_id','N/A')}")
            print(f" Abstract: {result['document'][:150]}..")
            print("=" * 60)
    except Exception as e:
        print(f"Error Searching: {e}")
    


def iterative_patent_exploration():
    ''' Performing Iterative Patent Exploration '''
    print("Iterative Patent Exploration")
    print("=" * 60)

    query = input("Enter the query to perform interative search on ...")
    if not query:
        print("Query cannot be empty")
        return
    
    steps = input("Enter the number of exploration steps (Default: 3):")
    try:
        steps = int(steps) if steps else 3
    except:
        steps = 3

    print(f"\n Exploring Patents related to {query} with {steps} refinement steps")

    try:
        results = iterative_search(query,steps)
        
        print(f"Found {len(results)} results through iterative search")
        print("=" * 60)
        for i,result in enumerate(results):
            meta = result["metadata"]
            print(f"{i+1}. {meta.get('title','N/A')}")
            print(f" Similarity Score: {round(1-result['distance'],4)}")
            print(f" Date: {meta.get('publication_date','N/A')}")
            print(f" Patent_ID: {meta.get('patent_id','N/A')}")
            print(f" Abstract: {result['document'][:150]}..")
            print("=" * 60)

    except Exception as e:
        print(F"Iterative Search Error: {e}")



def system_status():
    ''' Check the status of all system components '''
    # Check Chromadb Connection
    try:
        client = create_chromadb_client("localhost",8000)
        client.heartbeat()
        collections = client.list_collections()

        print("ChromaDB Connection: OK")
        print(f" Found {len(collections)} Collections")
        for col in collections:
            collection = client.get_collection(name=col.name)
            count = collection.count()
            print(f" {col.name}-{count} documents")
    except Exception as e:
        print(f"ChromaDB Connection: Failed - {e}")

    # Check Ollama API availability
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models",[])
            print("Ollama Connection: OK")
            print(f"Available Models: {','.join([m.get('name','unknown') for m in models])}")
        else:
            print(f"Ollama Connection: Failed - {response.status_code}")
    except Exception as e:
        print(f"Ollama Connection: Failed - {e}")

    # Check the Embedding Model
    try:
        sample = get_embedding("test")
        print(f"Embedding Model(nomic-embed-text): OK (Dimension: {len(sample)})")
    except Exception as e:
        print(f"Embedding Model: Failed - {e}")
    
    print("System is ready for operation.")
                                                    
def main():
    ''' Main Application Entry Point '''
    load_dotenv()

    while True:
        choice = display_main_menu()

        if choice == "1":
            run_complete_analysis()
        elif choice == "2":
            search_specific_patents()
        elif choice == "3":
            iterative_patent_exploration()
        elif choice == "4":
            system_status()
        elif choice == "5":
            print("\n Exiting Patent Innovation Predictor ")
            break
        else:
            print("\n Invalid Option. Try choosing different option between 1 to 5.")

        input("\n Press Enter to continue...")

if __name__ == "__main__":
    main()

