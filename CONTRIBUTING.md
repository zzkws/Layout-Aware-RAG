# Contributing

Thanks for helping improve traceable retrieval for engineering documents.

1. Create a focused branch from `main`.
2. Install development dependencies with `pip install -e ".[dev,embedding,mcp]"`.
3. Run `python -m pytest`, `python -m ruff check .`, and both site builds.
4. Do not commit API keys, model weights, generated full corpora, logs, absolute machine paths, or private network addresses.
5. Open a pull request describing the behavior change, evidence, tests, and any retrieval-baseline movement.

Changes that reduce the saved retrieval baseline must explain the tradeoff and include reviewed examples.
