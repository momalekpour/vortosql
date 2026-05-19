import pytest

from vortosql.core.database.adapters.base_adapter import BaseAdapter


def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        BaseAdapter()
