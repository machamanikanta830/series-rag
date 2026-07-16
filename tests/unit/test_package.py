"""Smoke tests for the initial package structure."""

from importlib import import_module


def test_app_package_is_importable() -> None:
    """The application package can be imported from the project root."""
    package = import_module("app")

    assert package.__name__ == "app"
