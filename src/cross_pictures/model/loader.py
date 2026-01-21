import onnxruntime as ort

def load_model(model_path: str, num_threads: int):
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session = ort.InferenceSession(model_path, session_options, providers=["CPUExecutionProvider"])
    return session


def load_labels(label_path: str) -> list[str]:
    with open(label_path, "r") as f:
        labels = [labels.strip() for labels in f.readlines()]
    return labels

