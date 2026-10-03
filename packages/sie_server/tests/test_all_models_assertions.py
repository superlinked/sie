"""Weightless regressions for the real-model suite's numerical xfail boundary."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from packages.sie_server.tests import test_all_models as models


@pytest.mark.parametrize("kind", ["dense", "sparse", "multivector"])
@pytest.mark.parametrize("defect", [None, "absent", "shape", "nonfinite", "numerical"])
def test_bge_m3_xfail_preserves_output_structure(
    monkeypatch: pytest.MonkeyPatch, kind: str, defect: str | None
) -> None:
    values = np.array([1.0, 2.0, 3.0])
    if defect == "numerical":
        values += 1.0
    elif defect == "nonfinite":
        values[0] = np.nan
    if kind == "dense":
        payload = None if defect == "absent" else values.reshape(1, 3)
        if defect == "shape":
            payload = values.reshape(3, 1)
        check = partial(models._assert_dense, "BAAI/bge-m3", 3, [1.0, 2.0, 3.0])
    elif kind == "sparse":
        payload = None if defect == "absent" else [SimpleNamespace(indices=np.array([1, 2, 3]), values=values)]
        if defect == "shape":
            payload[0].values = values[:2]
        check = partial(models._assert_sparse, "BAAI/bge-m3", [1, 2, 3], [1.0, 2.0, 3.0])
    else:
        payload = None if defect == "absent" else [values.reshape(1, 3)]
        if defect == "shape":
            payload = [values.reshape(3, 1)]
        check = partial(models._assert_multivector, "BAAI/bge-m3", 3, [1.0, 2.0, 3.0])
    output = SimpleNamespace(**{kind: payload})
    monkeypatch.setattr(models, "_get_adapter", lambda _: SimpleNamespace(encode=lambda *_args, **_kwargs: output))

    expected_failure = models._BGE_M3_CPU_BFLOAT16.kwargs["raises"]
    assert expected_failure is models._NumericalReferenceMismatchError
    assert models._BGE_M3_CPU_BFLOAT16.kwargs["strict"] is True
    if defect is None:
        check()
    elif defect == "numerical":
        with pytest.raises(expected_failure):
            check()
    else:
        with pytest.raises(AssertionError) as failure:
            check()
        assert not isinstance(failure.value, expected_failure)
