install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main tests/test_*.py

format:	
	black *.py 

lint:
	pylint --disable=R,C,pointless-string-statement --ignore-patterns=test_.*?py *.py 

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: install test format deploy
