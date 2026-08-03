from __future__ import annotations

import io
import sys
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from sie_server.adapters.st_sparse_vision.adapter import SparseEncoderVisionAdapter
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

IMMUTABLE_REV = "99bdc93f42460e595b2fb1e78b96edd44e898441"
VOCAB = 50368


def _png_bytes(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def _sparse_coo(batch: int) -> torch.Tensor:
    row_indices = torch.arange(batch).repeat_interleave(2)
    col_indices = torch.tensor([100, 500] * batch)
    indices = torch.stack([row_indices, col_indices])
    values = torch.full((2 * batch,), 0.5, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(batch, VOCAB))


class TestSparseEncoderVisionAdapter:
    """Tests for SparseEncoderVisionAdapter with mocked SparseEncoder."""

    @pytest.fixture(autouse=True)
    def _restore_syspath(self):
        """Undo any sys.path registration a load() performs (#2209)."""
        saved = list(sys.path)
        yield
        sys.path[:] = saved

    @pytest.fixture(autouse=True)
    def _restore_sysmodules(self):
        """Drop any custom module a load() imports or evicts (#2209)."""
        before = set(sys.modules)
        yield
        for name in set(sys.modules) - before:
            del sys.modules[name]

    @pytest.fixture(autouse=True)
    def mock_snapshot_download(self, tmp_path: Path) -> Iterator[MagicMock]:
        """Keep remote-path loads offline; return a stand-in snapshot dir."""
        with patch("sie_server.adapters.st_sparse_vision.adapter.snapshot_download") as m:
            m.return_value = str((tmp_path / "snapshot").resolve())
            yield m

    @pytest.fixture
    def mock_sparse_model(self) -> MagicMock:
        mock = MagicMock()
        mock.get_embedding_dimension.return_value = VOCAB

        def _encode(inputs, **kwargs):
            return _sparse_coo(len(inputs))

        mock.encode_query.side_effect = _encode
        mock.encode_document.side_effect = _encode
        return mock

    @pytest.fixture
    def adapter(self) -> SparseEncoderVisionAdapter:
        return SparseEncoderVisionAdapter(
            "test-model",
            max_seq_length=512,
            revision=IMMUTABLE_REV,
        )

    def test_capabilities(self, adapter: SparseEncoderVisionAdapter) -> None:
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["sparse"]

    def test_dims_before_load_returns_none(self, adapter: SparseEncoderVisionAdapter) -> None:
        assert adapter.dims.sparse is None

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")

        mock_class.assert_called_once_with(
            "test-model",
            device="cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "float32"},
            revision=IMMUTABLE_REV,
        )
        assert mock_sparse_model.max_seq_length == 512
        assert adapter.dims.sparse == VOCAB

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_precision_override(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
    ) -> None:
        """compute_precision flows into the checkpoint load dtype."""
        mock_class.return_value = mock_sparse_model
        adapter = SparseEncoderVisionAdapter(
            "test-model",
            compute_precision="float16",
            revision=IMMUTABLE_REV,
        )
        adapter.load("cpu")
        assert mock_class.call_args.kwargs["model_kwargs"] == {"torch_dtype": "float16"}

    def test_load_remote_code_requires_immutable_revision(self) -> None:
        adapter = SparseEncoderVisionAdapter("org/model", revision="main")
        with pytest.raises(ValueError, match="immutable 40-character revision"):
            adapter.load("cpu")

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_local_path_exempt_from_revision_gate(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter = SparseEncoderVisionAdapter(str(tmp_path))
        adapter.load("cpu")
        assert "revision" not in mock_class.call_args.kwargs

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_registers_remote_snapshot_on_syspath(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
    ) -> None:
        """Router resolves bare custom-module imports only if the snapshot dir
        is on sys.path; the load must register it for remote checkpoints (#2209).
        """
        mock_class.return_value = mock_sparse_model
        snapshot = mock_snapshot_download.return_value
        assert snapshot not in sys.path
        adapter.load("cpu")
        mock_snapshot_download.assert_called_once_with("test-model", revision=IMMUTABLE_REV)
        # Prepended so this checkpoint's custom module wins over any sibling's.
        assert sys.path[0] == snapshot

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_local_path_registers_local_dir_without_download(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter = SparseEncoderVisionAdapter(str(tmp_path))
        adapter.load("cpu")
        mock_snapshot_download.assert_not_called()
        assert sys.path[0] == str(tmp_path.resolve())

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_file_path_registers_parent_dir(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A file path resolves to its parent directory, not the file itself."""
        mock_class.return_value = mock_sparse_model
        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"")
        adapter = SparseEncoderVisionAdapter(str(model_file))
        adapter.load("cpu")
        mock_snapshot_download.assert_not_called()
        assert sys.path[0] == str(tmp_path.resolve())

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_evicts_stale_sibling_custom_module(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A same-named module cached from a different snapshot is evicted so this
        checkpoint's code is re-imported from its own directory (#2209).
        """
        mock_class.return_value = mock_sparse_model
        snapshot = tmp_path / "checkpoint"
        snapshot.mkdir()
        (snapshot / "modeling_st_vsplade.py").write_text("marker = 'this'\n")
        other = tmp_path / "sibling" / "modeling_st_vsplade.py"
        other.parent.mkdir()
        other.write_text("marker = 'other'\n")
        stale = ModuleType("modeling_st_vsplade")
        stale.__file__ = str(other)
        sys.modules["modeling_st_vsplade"] = stale

        adapter = SparseEncoderVisionAdapter(str(snapshot))
        adapter.load("cpu")

        assert "modeling_st_vsplade" not in sys.modules
        assert sys.path[0] == str(snapshot.resolve())

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_without_trust_remote_code_skips_registration(
        self,
        mock_class: MagicMock,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter = SparseEncoderVisionAdapter(str(tmp_path), trust_remote_code=False)
        before = list(sys.path)
        adapter.load("cpu")
        mock_snapshot_download.assert_not_called()
        assert sys.path == before

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_load_registers_snapshot_dir_only_once(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
        mock_snapshot_download: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        snapshot = mock_snapshot_download.return_value
        sys.path.insert(0, snapshot)
        adapter.load("cpu")
        assert sys.path.count(snapshot) == 1

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_encode_query_text(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["sparse"], is_query=True)

        assert output.batch_size == 2
        assert output.is_query is True
        assert output.sparse is not None
        assert len(output.sparse) == 2
        assert output.sparse[0].indices.tolist() == [100, 500]
        mock_sparse_model.encode_query.assert_called_once()
        assert mock_sparse_model.encode_query.call_args.args[0] == ["hello", "world"]
        mock_sparse_model.encode_document.assert_not_called()

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_encode_document_images(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [
            Item(images=[{"data": _png_bytes("red"), "format": "png"}]),
            Item(images=[{"data": _png_bytes("blue"), "format": "png"}]),
        ]
        output = adapter.encode(items, output_types=["sparse"], is_query=False)

        assert output.batch_size == 2
        assert output.is_query is False
        assert output.sparse is not None
        assert len(output.sparse) == 2
        mock_sparse_model.encode_document.assert_called_once()
        sent = mock_sparse_model.encode_document.call_args.args[0]
        assert len(sent) == 2
        assert all(isinstance(img, Image.Image) for img in sent)
        assert all(img.mode == "RGB" for img in sent)
        mock_sparse_model.encode_query.assert_not_called()

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_encode_uses_first_image_only(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Multi-image items follow the pipeline's one-image-per-item contract."""
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [
            Item(
                images=[
                    {"data": _png_bytes("red"), "format": "png"},
                    {"data": _png_bytes("blue"), "format": "png"},
                ]
            )
        ]
        output = adapter.encode(items, output_types=["sparse"], is_query=False)

        assert output.batch_size == 1
        sent = mock_sparse_model.encode_document.call_args.args[0]
        assert len(sent) == 1

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_rejects_dense_output(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode([Item(text="hello")], output_types=["dense"])

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_encode_without_text_or_image_raises(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")
        with pytest.raises(ValueError, match="requires text or image"):
            adapter.encode([Item()], output_types=["sparse"])

    def test_encode_before_load_raises(self, adapter: SparseEncoderVisionAdapter) -> None:
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.encode([Item(text="hello")], output_types=["sparse"])

    @patch("sie_server.adapters.st_sparse_vision.adapter.SparseEncoder")
    def test_unload(
        self,
        mock_class: MagicMock,
        adapter: SparseEncoderVisionAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        mock_class.return_value = mock_sparse_model
        adapter.load("cpu")
        assert adapter.dims.sparse == VOCAB
        adapter.unload()
        assert adapter.dims.sparse is None
