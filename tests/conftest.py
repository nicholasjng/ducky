import pytest

import ducky


@pytest.fixture
def con() -> ducky.Connection:
    """A fresh in-memory connection, isolated per test."""
    return ducky.connect()
