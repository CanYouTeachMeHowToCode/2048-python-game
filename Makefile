install:
	pip install --upgrade pip
	pip install -e "./2048[dev]"

test:
	cd 2048 && python -m pytest -vv --cov=game2048 test

format:
	black 2048/src 2048/test

lint:
	pylint --disable=R,C,pointless-string-statement,undefined-variable,no-member,function-redefined,unused-variable,unused-import,redefined-outer-name --ignore=UI.py --ignore-patterns=test_.*?py 2048/src/game2048

refactor: format lint

all: install test refactor
