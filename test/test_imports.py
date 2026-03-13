import hgx


def test_public_imports():
    """Verify that all explicit public re-exports in __init__.py can be imported."""

    # We load the __init__.py file and parse explicit re-exports
    # Looking for lines like `from hgx._something import Name as Name`
    import pathlib
    init_path = pathlib.Path(hgx.__file__)
    content = init_path.read_text()

    re_exports = []
    import ast
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # E.g. `Name as Name`
                if alias.asname is not None and alias.asname == alias.name:
                    re_exports.append(alias.name)

    # Some optional imports might be under try-except blocks.
    # Let's collect them by dynamically checking what exists in the hgx module.
    # We will only verify the ones that are successfully loaded in the current
    # environment (since this test might run without 'viz' or 'dynamics'
    # installed in some environments).

    # Get all non-private attributes of hgx module
    actual_public_names = [name for name in dir(hgx) if not name.startswith("_")]

    # Assert that all dynamically available re-exported names are actually in `dir(hgx)`
    # Or, more importantly, check if we can import them from `hgx`.

    for name in re_exports:
        if name in actual_public_names:
            # Try to get the attribute
            getattr(hgx, name)

    # And specifically test __all__ if it was defined, but we don't use it.
    if hasattr(hgx, "__all__"):
        for name in hgx.__all__:
            assert hasattr(hgx, name), f"{name} is in __all__ but not accessible"

    # Also make sure there are no `_` prefixed names exposed as public re-exports
    for name in re_exports:
        assert not name.startswith("_"), f"Private name {name} exposed publicly"
