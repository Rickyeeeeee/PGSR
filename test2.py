import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Initialize model, optimizer, and loss functions
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Generate dummy data
input_data = torch.randn(5, 10)
target_data_1 = torch.randn(5, 10)
target_data_2 = torch.randn(5, 10)

# Forward pass and calculate loss1 and loss2
output = model(input_data)
loss1 = loss_fn(output, target_data_1)

# --- Method 1: Three backward passes ---
# Zero gradients
optimizer.zero_grad()

# Perform the first backward pass (for loss1)
loss1.backward(retain_graph=True)  # Retain graph for second backward pass
gradients_after_loss1 = {name: param.grad.clone() for name, param in model.named_parameters()}

loss2 = loss_fn(output, target_data_2)
# Zero gradients again before computing loss2
optimizer.zero_grad()

# Perform the second backward pass (for loss2)
loss2.backward(retain_graph=True)  # Retain graph for the final backward pass
gradients_after_loss2 = {name: param.grad.clone() for name, param in model.named_parameters()}

# Zero gradients before the final backward pass
optimizer.zero_grad()

# Perform the third backward pass (for combined loss)
final_loss = loss1 + loss2
final_loss.backward(retain_graph=True) 
gradients_after_final_loss = {name: param.grad.clone() for name, param in model.named_parameters()}

# --- Method 2: Manual gradient accumulation ---
# Zero gradients
optimizer.zero_grad()

# Perform the first backward pass (for loss1)
loss1.backward(retain_graph=True)
gradients_loss1 = {name: param.grad.clone() for name, param in model.named_parameters()}

# Zero gradients again before computing loss2
optimizer.zero_grad()

# Perform the second backward pass (for loss2)
loss2.backward(retain_graph=True)
gradients_loss2 = {name: param.grad.clone() for name, param in model.named_parameters()}

# Manually accumulate gradients
for param, grad_loss1, grad_loss2 in zip(model.parameters(), gradients_loss1.values(), gradients_loss2.values()):
    param.grad = grad_loss1 + grad_loss2

# Perform parameter update step (just for comparison)
optimizer.step()

# Store gradients after manual accumulation
gradients_manual_accumulated = {name: param.grad.clone() for name, param in model.named_parameters()}

# --- Comparison ---
# Check if gradients after Method 1 and Method 2 are the same
for name in gradients_after_final_loss:
    assert torch.allclose(gradients_after_final_loss[name], gradients_manual_accumulated[name]), f"Gradients for {name} do not match!"

print("Gradients match! The manual gradient accumulation and the three backward passes produce identical results.")
