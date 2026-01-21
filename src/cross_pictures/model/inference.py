import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x)
    return e_x / e_x.sum()

def run_inference(session, input_tensor: np.ndarray, labels: list[str], top_k:int):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    probabilities = softmax(logits)
    top_indices = np.argsort(-probabilities)[:top_k][::-1]
    results = []
    for idx in top_indices:
        prediction = {
            "label": labels[idx],
            "confidence": round(float(probabilities[idx]),2),
        }
        results.append(prediction)
    return results
