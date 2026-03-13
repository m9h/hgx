# conda-forge feedstock recipe for hgx

## Submitting to conda-forge

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy the contents of this `conda-recipe/` directory into `recipes/hgx/` in your fork
3. Update the `sha256` hash in `meta.yaml`:
   - First publish hgx to PyPI (`git tag v0.1.0 && git push --tags` to trigger the release workflow)
   - Get the sha256 from PyPI: `pip hash --algorithm sha256 hgx-0.1.0.tar.gz`
   - Or from the PyPI JSON API: `curl -s https://pypi.org/pypi/hgx/0.1.0/json | python -m json.tool | grep sha256`
4. Open a PR against `conda-forge/staged-recipes`
5. The conda-forge CI will build and test the package automatically
6. Once merged, the `hgx` feedstock repo will be created at `conda-forge/hgx-feedstock`

## Notes

- The `sha256` in `meta.yaml` is set to `TODO` and **must** be updated after the first PyPI release
- The recipe uses `noarch: python` since hgx is a pure Python package
- Run dependencies match `pyproject.toml` exactly
