#Ejercicio Número 1
import torch as tr

tensor = tr.rand(2, 3)
print("Tensor en PyTorch:\n", tensor)

numpy_array = tensor.numpy()
print("\nConvertido a NumPy:\n", numpy_array)

#Ejercicio Número 2

A = tr.tensor([[1, 2], [3, 4]], dtype=tr.float32)
B = tr.tensor([[5, 6], [7, 8]], dtype=tr.float32)

print("Suma:\n", A + B)
print("Resta:\n", A - B)
print("Multiplicación:\n", A * B)

#Ejercicio Número 3

import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 4)  
        self.fc2 = nn.Linear(4, 1) 

    def forward(self, x):
        x = tr.relu(self.fc1(x))  
        x = tr.sigmoid(self.fc2(x))  
        return x

model = SimpleNN()
input_data = tr.rand(1, 3)
output = model(input_data)
print("Salida de la red neuronal:", output.item())