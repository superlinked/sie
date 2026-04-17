"""Tests for Haystack namespace alias exports."""

from haystack_integrations.components.embedders.sie import (
    SIEDocumentEmbedder as NamespacedDocumentEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIEImageEmbedder as NamespacedImageEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIEMultivectorDocumentEmbedder as NamespacedMultivectorDocumentEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIEMultivectorTextEmbedder as NamespacedMultivectorTextEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIESparseDocumentEmbedder as NamespacedSparseDocumentEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIESparseTextEmbedder as NamespacedSparseTextEmbedder,
)
from haystack_integrations.components.embedders.sie import (
    SIETextEmbedder as NamespacedTextEmbedder,
)
from haystack_integrations.components.extractors.sie import (
    Classification as NamespacedClassification,
)
from haystack_integrations.components.extractors.sie import (
    DetectedObject as NamespacedDetectedObject,
)
from haystack_integrations.components.extractors.sie import (
    Entity as NamespacedEntity,
)
from haystack_integrations.components.extractors.sie import (
    Relation as NamespacedRelation,
)
from haystack_integrations.components.extractors.sie import (
    SIEExtractor as NamespacedExtractor,
)
from haystack_integrations.components.rankers.sie import SIERanker as NamespacedRanker
from sie_haystack import (
    SIEDocumentEmbedder,
    SIEExtractor,
    SIEImageEmbedder,
    SIEMultivectorDocumentEmbedder,
    SIEMultivectorTextEmbedder,
    SIERanker,
    SIESparseDocumentEmbedder,
    SIESparseTextEmbedder,
    SIETextEmbedder,
)
from sie_haystack.extractors import Classification, DetectedObject, Entity, Relation


def test_namespaced_embedder_exports_match_flat_exports() -> None:
    assert NamespacedTextEmbedder is SIETextEmbedder
    assert NamespacedDocumentEmbedder is SIEDocumentEmbedder
    assert NamespacedSparseTextEmbedder is SIESparseTextEmbedder
    assert NamespacedSparseDocumentEmbedder is SIESparseDocumentEmbedder
    assert NamespacedMultivectorTextEmbedder is SIEMultivectorTextEmbedder
    assert NamespacedMultivectorDocumentEmbedder is SIEMultivectorDocumentEmbedder
    assert NamespacedImageEmbedder is SIEImageEmbedder


def test_namespaced_ranker_export_matches_flat_export() -> None:
    assert NamespacedRanker is SIERanker


def test_namespaced_extractor_exports_match_flat_exports() -> None:
    assert NamespacedExtractor is SIEExtractor
    assert NamespacedEntity is Entity
    assert NamespacedRelation is Relation
    assert NamespacedClassification is Classification
    assert NamespacedDetectedObject is DetectedObject
