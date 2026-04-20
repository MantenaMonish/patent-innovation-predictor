import os
import json
import uuid
import tiktoken
from embedding import get_embedding
from chromadb_client import create_chromadb_client, create_collection_if_not_exists

def load_patent_data(dir_path:str):
    '''
    Load patent data from the specified directory path.
    '''
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory path {dir_path} is not found.")
    
    all_files = os.listdir(dir_path)
    chunks=[]

    for file in all_files:
        if file.endswith(".json"):
            file_path = os.path.join(dir_path,file)
            with open(file_path,"r") as f:
                data = json.load(f)
            
            title = data.get("title")
            pdf = data.get("pdf")
            publication_date = data.get("publication_date")
            patent_id = data.get("search_parameters",{}).get("patent_id",None)
            abstract = data.get("abstract","")

            if not abstract or abstract.strip() == "":
                print(f"Skipping :{file} - empty abstract")
                continue

            token_count = len(
                tiktoken.encoding_for_model("gpt-3.5-turbo").encode(abstract)
            )
            embedding = get_embedding(abstract)

            if not embedding:
                print(f"Skipping {file} — empty embedding returned.")
                continue

            chunks.append({
                "title":title,
                "pdf":pdf,
                "publication_date":publication_date,
                "patent_id":patent_id,
                "abstract":abstract,
                "token_count":token_count,
                "embedding":embedding
            })
            
    print(f"Successfully prepared {len(chunks)} patents for indexing.")        
    return chunks

def index_patent_data(collection, patent_data:list):
    '''
    Upsert patent data into chromadb.
    Skips patents that are already indexed (by patent_id).
    Adds only new patents, preserving all existing data.
    '''
    # Get all IDs already in the collection
    existing_ids = set()
    if collection.count() > 0:
        existing = collection.get(include=[])  # fetch IDs only, no vectors
        existing_ids = set(existing["ids"])
        print(f"Collection already has {len(existing_ids)} patents indexed.")

    ids = []
    embeddings = []
    documents = []
    metadatas = []
    skipped = 0

    for patent in patent_data:
        patent_id = str(patent.get("patent_id") or uuid.uuid4())

        if patent_id in existing_ids:
            skipped += 1
            continue

        ids.append(patent_id)
        embeddings.append(patent["embedding"])
        documents.append(patent["abstract"])
        metadatas.append({
            "title":patent.get("title") or "",
            "pdf":patent.get("pdf") or "",
            "publication_date":patent.get("publication_date") or "",
            "patent_id":patent.get("patent_id") or "",
            "token_count":patent.get("token_count",0)
        })

    if skipped > 0:
        print(f"Skipped {skipped} patents already in the collection.")

    if not ids:
        print("No new patents to index — collection is already up to date.")
        return

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Indexed {len(ids)} new patents into '{collection.name}'. Total now: {collection.count()}.")

if __name__ == "__main__":
    dir_path = "files"
    host = "localhost"
    port = 8000
    
    client = create_chromadb_client(host,port)
    collection_name = "patents"
    collection = create_collection_if_not_exists(client,collection_name)

    try:
        patent_data = load_patent_data(dir_path)
        print(f"Loaded {len(patent_data)} from {dir_path}")
        index_patent_data(collection,patent_data)
    except Exception as e:
        print(f"Error: {e}")