.PHONY: proto openapi format ffmpeg

proto: 
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. ./api/grpc/proto/*.proto

## Importing the api package needs generated grpc code
openapi: proto
	python -m api.http.openapi

## FFmpeg DLLs for torchcodec, main.py picks them up from ./ffmpeg
ffmpeg:
	python fetch_ffmpeg.py

format:
	ruff check . --select I --select F401 --fix
	ruff format .