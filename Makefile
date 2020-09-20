.PHONY: format
format:
	python3 -m autopep8 **/*.py --in-place

.PHONY: lint
lint:
	python3 -m pylint **/*.py && python3 -m flake8 **/*.py --statistics
