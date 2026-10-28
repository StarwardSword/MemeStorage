from pathlib import Path
from typing import Any, cast, override

import librosa
import numpy as np
import onnxruntime as ort
from huggingface_hub import constants, hf_hub_download, snapshot_download
from PIL import Image
from transformers import AutoProcessor

from app import FileStorage

from .configuration import Configuration
from .gemma import DOCUMENT, QUERY, GemmaEmbeddings, Key

REPO = "onnx-community/embeddinggemma-2-ONNX"
# Plain files and not the symlinks of the hub cache: onnxruntime refuses external weights
# whose real path is outside the directory of the graph
LOCAL_DIR = Path(constants.HF_HOME) / "onnx" / "embeddinggemma-2"

# int8 weights, ~850 MB for all three encoders against ~1.5 GB of bfloat16.
# Measured on photos: cosine to the torch vectors is 0.998 and above
SUFFIX = "_quantized"

# Same prefixes as sentence-transformers uses for the torch model
PROMPTS = {QUERY: "task: search result | query: ", DOCUMENT: "title: none | text: "}

AUDIO_RATE = 16000
# Frames go through the vision encoder in groups, 32 at once would multiply its activations
VIDEO_FRAMES_PER_RUN = 8

FEATURES = ("image_features", "video_features", "audio_features")


class OnnxModel:
    """Speaks the part of the SentenceTransformer API that GemmaEmbeddings uses.

    The three encoders are separate graphs, so vision and audio are opened on the first
    file of that kind and a text-only server never pays for them
    """

    def __init__(self, config: Configuration) -> None:
        self.truncate_dim = config.truncate_dim
        self.options = ort.SessionOptions()
        self.options.inter_op_num_threads = 1
        # Every file has its own tensor shapes. With the arena and the memory pattern on, onnxruntime
        # keeps the biggest blocks it has ever needed: one video went from 2.7 GB to 1.3 GB without them
        self.options.enable_cpu_mem_arena = False
        self.options.enable_mem_pattern = False
        if config.threads is not None:
            self.options.intra_op_num_threads = config.threads

        # Everything but the weights: tokenizer, processor configs
        snapshot_download(
            REPO, allow_patterns=["*.json", "*.jinja"], local_dir=LOCAL_DIR
        )
        self.processor = AutoProcessor.from_pretrained(LOCAL_DIR)
        self.sessions: dict[str, ort.InferenceSession] = {}
        self._session("model")

    def _session(self, name: str) -> ort.InferenceSession:
        if name not in self.sessions:
            graph = f"onnx/{name}{SUFFIX}.onnx"
            path = hf_hub_download(REPO, graph, local_dir=LOCAL_DIR)
            # The weights live next to the graph in a second file
            hf_hub_download(REPO, f"{graph}_data", local_dir=LOCAL_DIR)
            self.sessions[name] = ort.InferenceSession(
                path, self.options, providers=["CPUExecutionProvider"]
            )
        return self.sessions[name]

    def _text_model(self, enc: Any, **features: np.ndarray) -> np.ndarray:
        feeds = {k: np.zeros((0, 512), np.float32) for k in FEATURES}
        feeds |= features
        feeds["input_ids"] = np.asarray(enc["input_ids"], dtype=np.int64)
        feeds["attention_mask"] = np.asarray(enc["attention_mask"], dtype=np.int64)
        return self._session("model").run(["sentence_embedding"], feeds)[0]

    def _vision(self, pixels: np.ndarray, positions: np.ndarray) -> np.ndarray:
        parts = [
            self._session("vision_encoder").run(
                ["image_features"],
                {
                    "pixel_values": pixels[i : i + VIDEO_FRAMES_PER_RUN],
                    "pixel_position_ids": np.asarray(
                        positions[i : i + VIDEO_FRAMES_PER_RUN], dtype=np.int64
                    ),
                },
            )[0]
            for i in range(0, len(pixels), VIDEO_FRAMES_PER_RUN)
        ]
        return np.concatenate(parts)

    def _media(self, item: dict[str, str]) -> np.ndarray:
        # Media goes in without a prompt: the model card says prefixes are for text only
        if Key.IMAGE in item:
            with Image.open(item[Key.IMAGE]) as img:
                enc = self.processor(images=[img.convert("RGB")], return_tensors="np")
            features = self._vision(enc["pixel_values"], enc["image_position_ids"])
            return self._text_model(enc, image_features=features)

        if Key.VIDEO in item:
            enc = self.processor(videos=[item[Key.VIDEO]], return_tensors="np")
            features = self._vision(
                enc["pixel_values_videos"], enc["video_position_ids"]
            )
            return self._text_model(enc, video_features=features)

        samples, _ = librosa.load(item[Key.AUDIO], sr=AUDIO_RATE, mono=True)
        enc = self.processor(audio=[samples], return_tensors="np")
        features = self._session("audio_encoder").run(
            ["audio_features"],
            {
                "input_features": enc["input_features"],
                "input_features_mask": enc["input_features_mask"],
            },
        )[0]
        return self._text_model(enc, audio_features=features)

    def encode(
        self, inputs: list[dict[str, str]], prompt_name: str | None, batch_size: int
    ) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(inputs)

        texts = [i for i, x in enumerate(inputs) if x.keys() == {Key.TEXT}]
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = self.processor(
                text=[
                    PROMPTS.get(prompt_name or "", "") + inputs[i][Key.TEXT]
                    for i in chunk
                ],
                return_tensors="np",
                padding=True,
            )
            for i, row in zip(chunk, self._text_model(enc)):
                rows[i] = row

        for i, item in enumerate(inputs):
            if rows[i] is None:
                rows[i] = self._media(item)[0]

        res = np.stack(cast(list[np.ndarray], rows))
        return res[:, : self.truncate_dim] if self.truncate_dim else res


class OnnxEmbeddings(GemmaEmbeddings):
    def __init__(self, config: Configuration, file_storage: FileStorage):
        if config.device != "cpu":
            raise ValueError("onnx backend runs on cpu only")
        super().__init__(config, file_storage)

    @override
    def _load(self) -> OnnxModel:
        return OnnxModel(self.config)
