import chromadb
from chromadb.config import Settings

def create_chromadb_client(host, port):
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False, is_persistent=True)
    )

    try:
        client.heartbeat()
        print("Connected to Chromadb")
        version = client.get_version()
        print(f"Chromadb Version: {version}")
    except Exception as e:
        print(f"Connection Failed: {e}")
        raise ConnectionError("Failed to connect to Chromadb") from e
    
    return client

def create_collection_if_not_exists(client, collection_name: str):
    '''
    Get existing Chromadb collection or create it with cosine similarity.
    Preserves all existing indexed patents — never deletes data.

    Args:
        client: Chromadb client instance
        collection_name: Name of the collection to get or create
    '''
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    count = collection.count()
    if count > 0:
        print(f"Found existing collection '{collection_name}' with {count} patents — preserving data.")
    else:
        print(f"Created new collection '{collection_name}'.")

    return collection

if __name__ == "__main__":
    host = "localhost"
    port = 8000
    client = create_chromadb_client(host, port)

    collections = client.list_collections()
    print("Available Collections:")
    for col in collections:
        print(f"  - {col.name}")