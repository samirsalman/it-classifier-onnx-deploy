import onnxruntime
import torch
from app.create_input import tokenize


class ModelState:
    def __init__(self):
        self.ort_session = onnxruntime.InferenceSession("/api/app/runtime/model.onnx")
        self.vocab = torch.load("/api/app/runtime/vocab.pth")
        self.vocab.set_default_index(0)

    def predict(self, text: str):
        input_name = self.ort_session.get_inputs()[0].name
        input_vector = tokenize(vocab=self.vocab, text=text)
        ort_inputs = {input_name: [input_vector.tolist()]}
        ort_outs = self.ort_session.run(None, ort_inputs)[0]
        ort_outs = torch.tensor(ort_outs)
        preds = torch.argmax(ort_outs, dim=1)
        return preds.item()
