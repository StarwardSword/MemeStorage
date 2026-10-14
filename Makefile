.PHONY: proto

## python -m grpc_tools.protoc -I. api/grpc/proto --python_out=api/grpc/proto --grpc_python_out=api/grpc/proto ./api/grpc/proto/*.proto

proto: 
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. ./api/grpc/proto/*.proto