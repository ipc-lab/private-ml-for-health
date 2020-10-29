from commons import get_model_from_global_agent

d_r_class = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


def get_prediction(input_tensor):
    model = get_model_from_global_agent()
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return d_r_class[predicted_idx]
