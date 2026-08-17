"""Opt-in integration checks against the pinned HTTP Qdrant server.

Run with ``RUN_QDRANT_INTEGRATION=1``.  The default unit suite remains fully
offline and uses Qdrant local mode, which cannot exercise server-side BM25
inference from ``Document`` values.
"""

from __future__ import annotations

import os
import uuid
from contextlib import suppress

import pytest
from qdrant_client import QdrantClient, models

from core.config import settings
from retrieval.scope import RetrievalScope
from retrieval.search.sparse import sparse_document

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="set RUN_QDRANT_INTEGRATION=1 to test the HTTP Qdrant server",
)


def test_native_bm25_and_scoped_hybrid_on_qdrant_server():
    """Exercise the real Document/options/IDF/RRF wire contract end to end."""
    collection = f"pr12_bm25_smoke_{uuid.uuid4().hex}"
    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        cloud_inference=True,
    )
    created = False
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
            sparse_vectors_config={
                settings.QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        created = True
        client.upsert(
            collection_name=collection,
            wait=True,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        "": [1.0, 0.0],
                        settings.QDRANT_SPARSE_VECTOR_NAME: sparse_document(
                            "quy định mật khẩu TC-456"
                        ),
                    },
                    payload={"dataset_id": "allowed", "content": "TC-456"},
                ),
                models.PointStruct(
                    id=2,
                    vector={
                        "": [0.0, 1.0],
                        settings.QDRANT_SPARSE_VECTOR_NAME: sparse_document(
                            "annual leave policy"
                        ),
                    },
                    payload={"dataset_id": "blocked", "content": "annual leave"},
                ),
            ],
        )

        query_filter = RetrievalScope(("allowed",)).qdrant_filter()
        sparse = client.query_points(
            collection_name=collection,
            query=sparse_document("TC-456"),
            using=settings.QDRANT_SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=2,
            with_payload=True,
        )
        hybrid = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=[1.0, 0.0],
                    filter=query_filter,
                    limit=2,
                ),
                models.Prefetch(
                    query=sparse_document("TC-456"),
                    using=settings.QDRANT_SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=2,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=61)),
            query_filter=query_filter,
            limit=2,
            with_payload=True,
        )

        assert [point.id for point in sparse.points] == [1]
        assert [point.id for point in hybrid.points] == [1]
    finally:
        if created:
            with suppress(Exception):
                client.delete_collection(collection)
        client.close()
