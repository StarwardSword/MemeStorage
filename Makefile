.PHONY: proto openapi format

proto: 
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. ./api/grpc/proto/*.proto

## Importing the api package needs generated grpc code
openapi: proto
	python -m api.http.openapi

format:
	ruff format .