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
    Create Chromadb collection with cosine similarity if it doesn't exist.

    Args:
        client: Chromadb client instance
        collection_name: Name of the collection to create
    '''
    existing = [col.name for col in client.list_collections()]

    if collection_name in existing:
        print(f"Deleting existing collection {collection_name}")
        client.delete_collection(name=collection_name)
    
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    print(f"Created Collection '{collection_name}'")
    return collection

if __name__ == "__main__":
    host = "localhost"
    port = 8000
    client = create_chromadb_client(host, port)

    collections = client.list_collections()
    print("Available Collections:")
    for col in collections:
        print(f"  - {col.name}")