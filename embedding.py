import requests

def get_embedding(prompt,model="nomic-embed-text"):
    '''
    To perform embedding on the provided prompt, while using the specified model.

    Args:
        prompt(str): The prompt to embed.
        model(str): Specifies the model that is to be used for embedding. The default model used here is "nomic-embed-text.
    
    Response:
        list: The embedding vector
    '''
    embed_url = "http://localhost:11434/api/embeddings"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "prompt": prompt
    }

    response = requests.post(embed_url,headers=headers,json=data)

    if response.status_code == 200: return response.json().get("embedding",[])
    else:
        raise Exception (f"Error fetching embeddings, status: {response.status_code} - {response.text}") 
    
if __name__ == "__main__":
    sample_prompt = "The Hussain Mohammed Bombed the neighbour country."
    try:
        result = get_embedding(sample_prompt)
        print(f"The Dimensions of the Embeddings: ",len(result))
        print(f"Embeddings: ", result)
    except Exception as e:
        print(f"Error fetching the embedding {e}")

