from pathlib import Path

import pytest

from memevo.algorithms import create_algorithm, register_algorithm
from memevo.datasets import create_dataset, register_dataset


def test_custom_algorithm_can_be_registered():
    expected = object()
    register_algorithm("test_algorithm", lambda models, path, settings: expected)

    assert create_algorithm("test_algorithm", object(), Path("."), {}) is expected


def test_custom_dataset_can_be_registered():
    expected = object()
    register_dataset("test_dataset", lambda settings: expected)

    assert create_dataset("test_dataset", {}) is expected


@pytest.mark.parametrize(
    ("create", "name"),
    [
        (lambda: create_algorithm("missing", object(), Path("."), {}), "algorithm"),
        (lambda: create_dataset("missing", {}), "dataset"),
    ],
)
def test_unknown_registry_entry_has_clear_error(create, name):
    with pytest.raises(ValueError, match=f"Unknown {name}"):
        create()
