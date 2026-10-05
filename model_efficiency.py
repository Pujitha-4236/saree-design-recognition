import time
import torch

from model import SareeEmbeddingModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create model
model = SareeEmbeddingModel(
    embedding_dim=256
)

model = model.to(device)
model.eval()


# Count parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("================================")
print("MODEL EFFICIENCY")
print("================================")

print(
    "Total parameters:",
    total_parameters
)

print(
    "Trainable parameters:",
    trainable_parameters
)

print(
    "Embedding dimension:",
    256
)


# Test input
input_image = torch.randn(
    1,
    3,
    224,
    224
).to(device)


# Warm-up
with torch.no_grad():

    for _ in range(5):
        model(input_image)


# Measure inference time
times = []

with torch.no_grad():

    for _ in range(20):

        start = time.perf_counter()

        model(input_image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(
            (end - start) * 1000
        )


average_latency = sum(times) / len(times)


print(
    "Device:",
    device
)

print(
    "Average inference latency:",
    f"{average_latency:.2f} ms"
)

print("================================")