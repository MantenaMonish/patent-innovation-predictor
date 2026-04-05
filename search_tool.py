from embedding import get_embedding
from chromadb_client import create_chromadb_client, create_collection_if_not_exists

collectionName = "patents"

def keyword_search(query_text,top_k=20):
    '''
    Perform keyword search by filering the metadata and documents.

    NOTE: ChromaDB doesn't have native full-text keyword search like OpenSearch's
    'match' query. We simulate it by using 'where_document' which checks if the
    document contains the query text.

    Args:
        query_text(str): The query text to search
        top_k(int): The number of results to return

    Returns:
        list: Search results
    '''
    try:
        collection = get_collection()
        query_embedding = get_embedding(query_text)
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where_document={"$contains": query_text},
            include = ["documents","metadatas","distances"]
        )
        return format_response(response)
        
    except Exception as e:
        print(f"Keyword Search Error {e}")
        return []

def semantic_search(query_text,top_k=20):
    '''
    Perform semantic search by using query embeddings

    Args:
        query_text(str): The query text to search
        top_k(int): Number of results to return

    Returns:
        list: Search results
    '''
    try:
        collection = get_collection()
        query_embedding = get_embedding(query_text)

        results = collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k,
            include = ["documents","metadatas","distances"]
        )

        return format_response(results)
    except Exception as e:
        print(f"Semantic Search Error: {e}")
        return []

def hybrid_search(query_text,top_k=20):
    '''
    Perform Hybrid Search by using both Keyword Search + Semantic Search

    NOTE: ChromaDB has no native hybrid search like OpenSearch's bool/should query.
    We manually merge semantic and keyword results, deduplicate by patent_id,
    and return the union — this is the standard approach with ChromaDB.

    Args:
        query_text(str): The query text to search
        top_k(int): Number of results to return

    Returns:
        list: Search Results
    '''
    try:
        semantic_results = semantic_search(query_text,top_k)
        keyword_results = keyword_search(query_text,top_k)

        seen_ids = set()
        merged = []

        for result in keyword_results + semantic_results:
            patent_id = result["metadata"].get("patent_id")
            if patent_id not in seen_ids:
                seen_ids.add(patent_id)
                merged.append(result)

        return merged[:top_k]
    except Exception as e:
        print(f"Hybrid Search Error: {e}")

        try:
            return semantic_search(query_text,top_k)
        except Exception as e1 :
            print(f"Fallback search error: {e1}")
            return []

def get_collection():
    '''
    Helper to get patent collection
    '''
    client = create_chromadb_client("localhost",8000)
    return client.get_collection(name=collectionName)

def iterative_search(query_text,refinement_steps=3, top_k=20):
    '''
    Perform iterative sarch with query refinement.

    Args:
        query_text(str): The initial query text
        refinement_steps(int): Number of search refinements iterations
        top_k(int): Number of results per iterations
    
        Returns:
            list: Search results
    '''
    all_results = []
    seens_ids = set()
    current_query = query_text
    
    for i in range(refinement_steps):
        try:
            results = semantic_search(query_text,top_k)

            for result in results:
                patent_id = result["metadata"].get("patent_id")
                if patent_id not in seens_ids:
                    seens_ids.add(patent_id)
                    all_results.append(result)
            if not results:
                break

            top_result = results[0]
            current_query = f"{current_query} {top_result['metadata']['title']}"

        except Exception as e:
            print(f"Iterative seach error at step {i}: {e}")
            break
        
    return all_results


def format_response(response):
    '''
    Format Chromadb raw response into clean format

    ChromaDB returns parallel lists (ids, documents, metadatas, distances),
    this zips them into one readable structure.

    Args:
        response: raw chromadb query response

    Returns:
        A list of dicts with keys: ids, documents, metadata, distance
    '''
    results = []

    ids = response.get("ids",[[]])[0]
    documents = response.get("documents",[[]])[0]
    metadatas = response.get("metadatas",[[]])[0]
    distances  = response.get("distances",[[]])[0]

    for id,doc,meta,dist in zip(ids,documents,metadatas,distances):
        results.append({
            "id":id,
            "document":doc,
            "metadata":meta,
            "distance":dist
        })

    return results

if __name__ == "__main__":
    query = "lithium battery"

    print("\nSemantic Search Results:")
    semantic_results = semantic_search(query)
    for res in semantic_results:
        print(f"Title: {res['metadata']['title']}, Patent ID: {res['metadata']['patent_id']}")
        print(f"Distance: {res['distance']}")
        print()

    print("\nHybrid Searc Results:")
    hybrid_results = hybrid_search(query)
    for res in hybrid_results:
        print(f"Title: {res['metadata']['title']}, Patent ID: {res['metadata']['patent_id']}")
        print()