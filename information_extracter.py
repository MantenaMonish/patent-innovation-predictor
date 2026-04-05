import os
import requests
import json
import pprint
from dotenv import load_dotenv
from helper import get_data_from_serpapi,get_serpapi_url

load_dotenv()

def fetch_patent_data(query,dir_path):
    '''
    Fetch Patent Data and Store it in the given directory path.

    Args:
        query(str): Search Query for the Patents.
        dir_path (str): The directory to save the results.
    '''
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        raise ValueError("SERP_API_KEY environment variable is not set.")
    
    os.makedirs(dir_path,exist_ok=True)

    url=f"https://serpapi.com/search?engine=google_patents&q={query}&api_key={api_key}"

    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("")
        exit(1)

    if response.status_code == 200:
        data = response.json()
        for idx,patent in enumerate(data.get("organic_results",[])):
            serp_url = get_serpapi_url(patent)
            patent_data = get_data_from_serpapi(serp_url)
            if not patent_data:
                print(f"Error fetching data from the patn index:{idx}.") 
                continue
            
            with open(f"{dir_path}/patent_data_{idx}.json","w") as file:
                json.dump(patent_data,file,indent=2)
    else:
        raise ValueError(f"Error: {response.status_code},{response.text}")
    
if __name__ == "__main__" :
    query = input("Enter the search query for the patents to be fetched:")
    dir_path = input("Enter the directory path you want the fetched results to be stored:")
    try:
        fetch_patent_data(query,dir_path)
        print(f"The Result is Saved to the Path : {dir_path}.")
    except Exception as e:
        print(f"Exception: {e}.")
