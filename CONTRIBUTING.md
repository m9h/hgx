# Contributing

Contributions are welcome! This guide follows the same workflow used by
[diffrax](https://github.com/patrick-kidger/diffrax) and other JAX ecosystem
libraries.

## Getting started

1. **Fork** this repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hgx.git
   cd hgx
   ```
3. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Running tests

```bash
pytest
```

## Submitting changes

1. Create a new branch for your work:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes and commit them.
3. Push your branch to your fork:
   ```bash
   git push origin my-feature
   ```
4. Open a **Pull Request** against `main` on this repository.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Pre-commit hooks will run ruff automatically, but you can also run
it manually:

```bash
ruff check hgx/ test/
```
