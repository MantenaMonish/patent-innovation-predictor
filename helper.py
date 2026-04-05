import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SERP_API_KEY")
if not api_key:
    raise ValueError("SERP_API_KEY environment variable is not set.")

def get_serpapi_url(data):
    '''
    To construct a SerpApi link from provided data.
    
    Args:
        data (dict): To Fetch serpapi link from the given data.
    
    Returns:
        str: The complete SerpApi link with the API key.

    '''
    if "serpapi_link" not in data:
        raise ValueError("The Serpapi link does not exist in the provided data")
    
    serpapi_url = data["serpapi_link"]

    if "api_key=" not in serpapi_url:
        separator = "&" if "?" in serpapi_url else "?"
        serpapi_url =f"{serpapi_url}{separator}api_key={api_key}"
    
    return serpapi_url


def get_data_from_serpapi(serpapi_url):
    '''
    Fetches Data From the provided Serp Api Url.

    Args: 
        serapi_url (str): To fetch data from.

    Returns: 
        dict: The parsed json response from SerpAPI.
    
    Raises:
        HttpError: Incase the HttRequest raises an error status code.
    '''
    params = {
        "api_key" : api_key
    }
    response = requests.get(serpapi_url,params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return response.raise_for_status()