import torch

# Original tensor
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Original shape:", x.shape)  

# Adding a new dimension at index 0
y = torch.unsqueeze(x, 0)
print("Shape after unsqueeze at dim 0:", y.shape)  
print(y)
# Adding a new dimension at index 1
z = torch.unsqueeze(x, 1)
print("Shape after unsqueeze at dim 1:", z.shape)  
print(z)
# Adding a new dimension at index -1 (last position)
w = torch.unsqueeze(x, -1)
print("Shape after unsqueeze at dim -1:", w.shape)  
print(w)