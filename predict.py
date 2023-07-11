from cog import BasePredictor, Path, Input
import torch
import onnx
import numpy as np
import onnxruntime
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from torch.nn import functional as F


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self,
                prompt: str = Input(description="prompt")
                ) -> list:
        self.model_name = './EleutherAI-gpt-neo-1.3B-5gb-onnx/1/model.onnx'
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "EleutherAI/gpt-neo-1.3B")
        tokens = self.tokenizer(prompt, return_tensors="np")
        ort_session = onnxruntime.InferenceSession(self.model_name)
        label_name = ort_session.get_outputs()[0].name
        ort_outs = ort_session.run([label_name], dict(tokens))
        return ort_outs
