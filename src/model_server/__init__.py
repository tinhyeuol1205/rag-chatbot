"""Self-hosted embedding and reranking service for Apple Silicon deployments.

The package intentionally keeps model imports lazy.  Importing the API (for
example while running unit tests or generating OpenAPI) must not download BGE
weights or initialise an MPS context.
"""

