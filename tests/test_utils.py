"""Tests for `faster_whisper_rocm.utils`.

Currently a smoke test for the utils package; the deprecated `greet` helper
and related `hello` command were removed.
"""


def test_utils_package_smoke() -> None:
    """Tests that the utils package can be imported and has a docstring."""
    import faster_whisper_rocm.utils as utils
    import faster_whisper_rocm.utils.helpers as helpers  # noqa: F401

    # Package should import and have a module docstring
    assert getattr(utils, "__doc__", None) is not None
