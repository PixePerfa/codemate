"""General util for querying different types of open source vectors DB."""

import chromadb
import openai
import uuid
import os
from typing import Dict, Any, List, Optional, cast
from enum import Enum
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
)


class DBConfig(BaseModel):
    name: str
    path: str
    collection_name: str
    emb_name: str


class DBSUPPORTED(Enum):
    CHROMA = 0
    QDRANT = 1


class EMBSUPPORTED(Enum):
    DEFAULT = 0
    OPENAI = 1


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._dimension = 64

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._client.embeddings.create(
            input=input, model="text-embedding-3-small", dimensions=self._dimension
        ).data
        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e.index)

        # Return just the embeddings
        return cast(Embeddings, [result.embedding for result in sorted_embeddings])

    @property
    def dimension(self):
        return self._dimension


class VectorDB:
    def __init__(self, config: DBConfig):
        self._config = config
        self._is_chroma_db = self._config.name == DBSUPPORTED.CHROMA.name
        self._is_qdrant_db = self._config.name == DBSUPPORTED.QDRANT.name
        self._collection_name = self._config.collection_name
        self._emb_func = self._get_emb_func()
        if self._is_qdrant_db:
            assert self._emb_func is not None, "Require a valid emb func for Qdrant DB."
        self._make_collection_instance()

    def _get_emb_func(self) -> Optional[Any]:
        if self._config.emb_name == EMBSUPPORTED.OPENAI.name:
            return OpenAIEmbeddingFunction()

    def _make_collection_instance(self):
        if self._is_chroma_db:
            self._client = chromadb.PersistentClient(
                path=self._config.path,
            )
            self._chroma_collection = self._client.get_or_create_collection(
                self._collection_name,
                embedding_function=self._emb_func
                if self._emb_func is not None
                else DefaultEmbeddingFunction(),
            )
        elif self._is_qdrant_db:
            self._client = QdrantClient(
                url=self._config.path,
                api_key=os.environ.get("QDRANT_API_KEY", None),
            )
            if not self._client.collection_exists(
                collection_name=self._collection_name
            ):
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self._emb_func.dimension,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                    shard_number=4,
                )
        else:
            raise ValueError(f"DB Type {self._config.name} not supported!")

    @property
    def client(self):
        return self._client

    def filter(self, kvs: Dict[str, str]) -> Optional[Any]:
        """Metadata / Payload filtering using kvs"""

        def _convert_to_chroma_op(results):
            ids = []
            metadatas = []
            documents = []
            for res in results:
                ids.append(res.id)
                metadatas.append(res.payload)
                documents.append(res.payload.get("document", None))
            return {
                "ids": ids,
                "metadatas": metadatas,
                "documents": documents,
            }

        if self._is_chroma_db:
            print({key: {"$eq": val} for key, val in kvs.items()})
            return self._chroma_collection.get(
                where={key: {"$eq": val} for key, val in kvs.items()}
            )
        elif self._is_qdrant_db:
            result = self.client.scroll(
                collection_name=self._collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=val),
                        )
                        for key, val in kvs.items()
                    ]
                ),
                with_payload=True,
            )[0]
            return _convert_to_chroma_op(result)

    def search(self, query: List[Any], top_k: int) -> Optional[Any]:
        """Find nearest neighbors for a given list of query."""

        def _convert_to_chroma_op(results):
            ids = []
            distances = []
            metadatas = []
            documents = []
            for result in results:
                ids.append([record.id for record in result[0]])
                metadatas.append([record.payload for record in result[0]])
                documents.append(
                    [record.payload.get("document", None) for record in result[0]]
                )
                distances.append(result[1])
            return {
                "ids": ids,
                "metadatas": metadatas,
                "documents": documents,
                "distances": distances,
            }

        if self._is_chroma_db:
            return self._chroma_collection.query(
                query_embeddings=query
                if all(isinstance(x, list) for x in query)
                and all(isinstance(x, float) for x in query[0])
                else None,
                query_texts=query if all(isinstance(x, str) for x in query) else None,
                n_results=top_k,
            )
        elif self._is_qdrant_db:
            # get embeddings if queries are strings.
            if all(isinstance(x, list) for x in query) and all(
                isinstance(x, float) for x in query[0]
            ):
                return self._client.search_batch(
                    collection_name=self._collection_name,
                    requests=[
                        qdrant_models.SearchRequest(vector=vec, limit=top_k)
                        for vec in query
                    ],
                )
            elif all(isinstance(x, str) for x in query):
                query_emb = self._emb_func(query)
                results = self._client.search_batch(
                    collection_name=self._collection_name,
                    requests=[
                        qdrant_models.SearchRequest(vector=vec, limit=top_k)
                        for vec in query_emb
                    ],
                )
                results_with_payload = []
                for result in results:
                    ids = [point.id for point in result]
                    scores = [point.score for point in result]
                    payloads = self._client.retrieve(
                        collection_name=self._collection_name,
                        ids=ids,
                        with_payload=True,
                    )
                    results_with_payload.append((payloads, scores))
                return _convert_to_chroma_op(results_with_payload)

    def add(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[Any],
    ):
        """Batch upsert documents to the collection"""
        if self._is_chroma_db:
            return self._chroma_collection.add(
                documents=documents, metadatas=metadatas, ids=list(map(str, ids))
            )
        elif self._is_qdrant_db:
            # openai has rate limit when embedding more than 1000 documents.
            embeddings = []
            for i in range(0, len(documents), 1000):
                embeddings.extend(self._emb_func(documents[i : i + 1000]))
            self._client.upsert(
                collection_name=self._collection_name,
                points=qdrant_models.Batch(
                    ids=[
                        str(uuid.uuid5(uuid.NAMESPACE_DNS, str(x)))
                        if not isinstance(x, int)
                        else x
                        for x in ids
                    ],
                    payloads=[
                        {**metadata, **{"document": document}}
                        for metadata, document in zip(metadatas, documents)
                    ],
                    vectors=embeddings,
                ),
            )
