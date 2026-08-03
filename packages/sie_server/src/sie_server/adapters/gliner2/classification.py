"""Transformers-5 routing seam for GLiNER2 classification models."""

from sie_server.adapters.gliner2.adapter import GLiNER2Adapter


class GLiNER2ClassificationAdapter(GLiNER2Adapter):
    """Route GLiNER2 classifiers independently from default-bundle NER models."""
