import torch
import torch.nn as nn


# Set the random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)  # Set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using CUDA)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results
    torch.backends.cudnn.benchmark = False  # Disable the autotuner to avoid non-deterministic behavior

# Define a simple neural network with an additional layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)  # Additional hidden layer
        self.fc3 = nn.Linear(3, 1)


    def forward(self, x):
        x = self.fc1(x)
        hidden = self.fc2(x)  # Hidden layer
        output = self.fc3(hidden)
        return output, hidden

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # Use a custom initialization, for example, Xavier uniform
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
# Input data shared across all cases
input_data = torch.randn(10)

def case1_without_retain_grad(input_data):
    """Case 1: Compute gradients without retain_grad"""
    set_random_seed(42)
    model = SimpleNN()  # New model instance
    loss_fn = nn.MSELoss()
    target1 = torch.tensor([1.0])
    target2 = torch.tensor([0.0])

    output1, hidden1 = model(input_data)
    output2, hidden2 = model(input_data)
    loss1 = loss_fn(output1, target1)
    loss2 = loss_fn(output2, target2)

    # Backward for loss1
    loss1.backward(retain_graph=True)
    grad_loss1 = model.fc1.weight.grad.clone()
    
    # Zero gradients
    # for param in model.parameters():
    #     param.grad = None

    # Backward for loss2
    loss2.backward()
    grad_loss2 = model.fc1.weight.grad.clone()

    print(loss1 + loss2)
    return grad_loss1, grad_loss2

def case2_with_retain_grad(input_data):
    """Case 2: Compute gradients without retain_grad"""
    set_random_seed(42)
    model = SimpleNN()  # New model instance
    loss_fn = nn.MSELoss()
    target1 = torch.tensor([1.0])
    target2 = torch.tensor([0.0])

    output1, hidden1 = model(input_data)
    output2, hidden2 = model(input_data)

    hidden1.retain_grad()

    loss1 = loss_fn(output1, target1)
    loss2 = loss_fn(output2, target2)

    # Backward for loss1
    loss1.backward()
    grad_loss1 = model.fc1.weight.grad.clone()
    
    # Zero gradients
    for param in model.parameters():
        param.grad = None

    # Backward for loss2
    loss2.backward()
    grad_loss2 = model.fc1.weight.grad.clone()

    print(loss1 + loss2)
    return grad_loss1 + grad_loss2

def case3_combined_loss(input_data):
    """Case 3: Compute gradients using a combined loss"""
    set_random_seed(42)
    model = SimpleNN()  # New model instance
    loss_fn = nn.MSELoss()
    target1 = torch.tensor([1.0])
    target2 = torch.tensor([0.0])

    output1, hidden1 = model(input_data)
    output2, hidden2 = model(input_data)
    loss1 = loss_fn(output1, target1)
    loss2 = loss_fn(output2, target2)
    loss = loss1 + loss2

    loss.backward()
    grad_case3 = model.fc1.weight.grad.clone()
    print(loss)
    return grad_case3

# Run each case and compare results
grad_loss1_case1, grad_loss2_case1 = case1_without_retain_grad(input_data)
grad_case2 = case2_with_retain_grad(input_data)
grad_case3 = case3_combined_loss(input_data)

# Print results
print("Gradient Case 1 Loss 1 (Without retain_grad):", grad_loss1_case1)
print("Gradient Case 1 Loss 2 (Without retain_grad):", grad_loss2_case1)
print("Gradient Case 2 (With retain_grad):", grad_case2)
print("Gradient Case 3 (loss1 + loss2):", grad_case3)
print("Are Case 1 (Loss 1 + Loss 2) and Case 2 gradients equal?", torch.allclose(grad_loss1_case1 + grad_loss2_case1, grad_case2))
print("Are Case 1 (Loss 1 + Loss 2) and Case 3 gradients equal?", torch.allclose(grad_loss1_case1 + grad_loss2_case1, grad_case3))
# print("Are Case 2 and Case 3 gradients equal?", torch.allclose(grad_case2, grad_case3))
