"""Haystack namespace exports for SIE embedders.

This mirrors Haystack's `haystack_integrations.components.*` convention
while keeping the existing `sie_haystack` imports available.
"""

from sie_haystack.embedders import (
    SIEDocumentEmbedder,
    SIEImageEmbedder,
    SIEMultivectorDocumentEmbedder,
    SIEMultivectorTextEmbedder,
    SIESparseDocumentEmbedder,
    SIESparseTextEmbedder,
    SIETextEmbedder,
)

__all__ = [
    "SIEDocumentEmbedder",
    "SIEImageEmbedder",
    "SIEMultivectorDocumentEmbedder",
    "SIEMultivectorTextEmbedder",
    "SIESparseDocumentEmbedder",
    "SIESparseTextEmbedder",
    "SIETextEmbedder",
]
