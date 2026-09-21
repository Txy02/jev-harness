import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless TYPESAFE_API_KEY is set."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    skip = pytest.mark.skip(reason="TYPESAFE_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
