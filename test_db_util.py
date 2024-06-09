import os
import pdb
from typing import List, Any
from db_util import VectorDB, DBConfig


def _create_test_docs_with_metadata() -> List[Any]:
    return [
        ("LlamaIndex is a data framework for your LLM applications", {
            "repo_url": "https://github.com/run-llama/llama_index"
        }),
        ("LangChain is a framework for developing applications powered by large language models (LLMs).",
         {
             "repo_url": "https://github.com/langchain-ai/langchain"
         })
    ]

def test_chromadb():
    config = DBConfig(
        name="CHROMA",
        path="/tmp",
        collection_name="db_util_test",
        emb_name="DEFAULT"
    )
    vector_db = VectorDB(config)
    assert vector_db.client
    docs_with_metadata = _create_test_docs_with_metadata()
    docs = [doc[0] for doc in docs_with_metadata]
    metadata  = [doc[1] for doc in docs_with_metadata]
    ids = [i for i,_ in enumerate(docs)]
    
    vector_db.add(docs, metadata, ids)
    
    filter_op = vector_db.filter(kvs={
        "repo_url": "https://github.com/run-llama/llama_index"
    })
    print("chroma filter op", filter_op)
    assert len(filter_op["documents"]) > 0
    search_op = vector_db.search(query=["Framework for developing LLM applications for coding"], top_k=2)
    assert len(search_op["documents"][0]) > 1
    print("chroma search op", search_op)



def test_qdrant():
    config = DBConfig(
        name="QDRANT",
        path="https://b72e9ec5-8c05-4475-a387-d84fbbee7460.us-east4-0.gcp.cloud.qdrant.io:6333",
        collection_name="db_util_test",
        emb_name="OPENAI"
    )
    vector_db = VectorDB(config)
    assert vector_db.client

    docs_with_metadata = _create_test_docs_with_metadata()
    docs = [doc[0] for doc in docs_with_metadata]
    metadata  = [doc[1] for doc in docs_with_metadata]
    ids = [i for i,_ in enumerate(docs)]
    vector_db.add(docs, metadata, ids)

    # filter results based on metadata
    filter_op = vector_db.filter(kvs={
        "repo_url": "https://github.com/run-llama/llama_index"
    })
    assert len(filter_op["documents"]) > 0
    print("qdrant filter op", filter_op)

    search_op = vector_db.search(query=["Framework for developing LLM applications for coding"], top_k=2)
    assert len(search_op["documents"][0]) == 2
    print("qdrant search op", search_op)

    


if __name__ == "__main__":
    test_chromadb()
    test_qdrant()
