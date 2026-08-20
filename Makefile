.PHONY: install test

install:
	python3 -m pip install -e .

test:
	pytest

# Example: make inspect CACHE=/path/to/cache
inspect:
	python3 scripts/inspect_cache.py --cache "$(CACHE)"
