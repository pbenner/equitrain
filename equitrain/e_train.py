import torch
import time
from torch.utils.data import DataLoader
from lightning.pytorch.demos import WikiText2, Transformer  # Keeping the same model and dataset as in train_fabric.py
import tracemalloc  # For CPU memory tracking

# Set the device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset and model
dataset = WikiText2()
dataloader = DataLoader(dataset)
model = Transformer(vocab_size=dataset.vocab_size).to(device)

# Set up the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Performance tracking variables
training_losses = []
epoch_times = []
total_start_time = time.time()
tracemalloc.start()  # Start memory tracking

# Training loop
model.train()
for epoch in range(20):
    epoch_start_time = time.time()

    epoch_loss = 0
    for batch in dataloader:
        input_data, target_data = batch
        input_data, target_data = input_data.to(device), target_data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data, target_data)

        # Compute loss
        loss = torch.nn.functional.nll_loss(output, target_data.view(-1))
        epoch_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Track performance
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

    training_losses.append(epoch_loss / len(dataloader))

    # Memory usage tracking
    current_allocated_memory = torch.cuda.memory_allocated(device) / 1e6  # in MB
    current_reserved_memory = torch.cuda.memory_reserved(device) / 1e6  # in MB
    print(f"Epoch [{epoch+1}/20] - Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s - Allocated Memory: {current_allocated_memory:.2f}MB - Reserved Memory: {current_reserved_memory:.2f}MB")

# Total training time
total_training_time = time.time() - total_start_time
print(f"Total Training Time: {total_training_time:.2f}s")

# Memory stats
current, peak = tracemalloc.get_traced_memory()
print(f"Peak CPU Memory Usage: {peak / 10**6:.2f}MB")
tracemalloc.stop()

# Final summary of results
print(f"Average Time per Epoch: {sum(epoch_times) / len(epoch_times):.2f}s")
print(f"Final Training Loss: {training_losses[-1]:.4f}")