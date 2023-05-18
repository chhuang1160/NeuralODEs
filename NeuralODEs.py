import csv
import gc
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Implement a simple ODE solver using the midpoint method 
def ode_solver(z0, func, t0, t1):
    step_num =  math.ceil(abs(t1 - t0) / 0.1)
    h = (t1 - t0) / step_num  

    z0_is_tuple = isinstance(z0, tuple)
    if z0_is_tuple:
        z0, in_shapes = flatten_tuple(z0)
        func = FuncFlattener(func, in_shapes)

    z = z0
    t = t0
    for i in range(step_num):

        z_mid = z + func(z, t) * (h / 2)
        z = z + func(z_mid, t + h / 2) * h 
        t = t + h 

    if z0_is_tuple:  
        z = unflatten_tensor(z, in_shapes)

    return z


def unflatten_tensor(z, in_shapes):
    # Unflattens a tensor z into a tuple of tensors based on in_shapes
    split_sections = [np.prod(shape) for shape in in_shapes]
    z = tuple(
        tensor.reshape(shape) for tensor, shape \
            in zip(torch.split(z, split_sections), in_shapes)
        )
    return z


def flatten_tuple(z):
    # Flattens a tuple of tensors and returns a 1D tensor and the input shapes
    in_shapes = tuple(tensor.shape for tensor in z)
    z = torch.cat([tensor.reshape(-1) for tensor in z])
    return z, in_shapes


class FuncFlattener(nn.Module):
    # Wraps a tuple function to flatten its input and output
    def __init__(self, func, in_shapes):
        super().__init__()
        self.func = func
        self.in_shapes = in_shapes

    def forward(self, z, t):
        z = unflatten_tensor(z, self.in_shapes)
        output, _ = flatten_tuple(self.func(z, t))
        return output
    

# Use torch.autograd.Function to implement a custom backward pass
# Forward pass: solve an ODE and return solutions for a given time sequence t 
# Backward pass: implement the adjoint sensitivity algorithm 
class AdjointMethod(torch.autograd.Function):  

    @staticmethod
    def forward(ctx, z0, func, t, *func_params):
        with torch.no_grad():
            z = torch.empty(len(t), *z0.shape).to(z0)
            z[0] = z0
            for i in range(len(t) - 1):
                z[i+1] = ode_solver(z[i], func, t[i], t[i+1])

        ctx.func = func  
        ctx.save_for_backward(z, t, *func_params)
        return z  

    @staticmethod
    def backward(ctx, grad_L_z):
        # grad_L_z has shape (len(t), batch_size, *input_shape)
        with torch.no_grad():
            func = ctx.func 
            z, t, *func_params = ctx.saved_tensors
            func_params = tuple(func_params)
            N = len(t) - 1  

            def aug_dynamics(z_aug, t):
                z = z_aug[0]
                a = z_aug[1]
                
                # Compute vector-Jacobian products
                with torch.enable_grad():
                    z = z.detach().requires_grad_(True)
                    t = t.detach()         
                    inputs = (z, *func_params)
                    outputs = func(z, t)
                    
                    vec_jac_prods = torch.autograd.grad(
                        outputs=outputs, inputs=inputs, 
                        grad_outputs=-a, allow_unused=True, retain_graph=True
                        )

                # Set vector-Jacobian products to 0 if autograd.grad returns None
                vec_jac_prods = [
                    vec_jac_prod if vec_jac_prod is not None else torch.zeros_like(input) 
                    for vec_jac_prod, input in zip(vec_jac_prods, inputs)
                    ] 
                
                return (outputs, *vec_jac_prods)

            # Solve reverse-time ODEs
            a = grad_L_z[N]
            grad_params = tuple(torch.zeros_like(param) for param in func_params)
            
            for i in range(N):
                s0 = (z[N-i], a, *grad_params)
                s = ode_solver(s0, aug_dynamics, t[N-i], t[N-i-1]) 
                
                # Unpack the solution and adjust the adjoint state 
                a = s[1] 
                grad_params = s[2:]
                a += grad_L_z[N-i-1]
    
        return a, None, None, *grad_params  
        # Outputs correspond to the inputs of the forward pass  
        

class ODENetwork(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, z0, t):
        z = AdjointMethod.apply(
            z0, self.func, t, *self.func.parameters()
            )
        return z


# Experiment with the performance on MNIST when using the adjoint method 
# and when directly backpropagating through the ODE solver
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
dataset = datasets.MNIST(
    root='data/mnist', train=True, download=True, transform=transform
    )
test_dataset = datasets.MNIST(
    root='data/mnist', train=False, download=True, transform=transform
    )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# Set up a class of neural networks to parametrize the dynamics of hidden states
class ODEFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(65, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z, t):
        t = t.expand(z[:, :1, :, :].shape) 
        z = self.conv1(torch.cat([z, t], dim=1))
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(torch.cat([z, t], dim=1))
        z = self.bn2(z)
        z = self.relu(z)
        return z


# Define MNIST classifiers consisting of downsampling layers,
# feature layers using ODE networks, and fully connected layers 
class MNISTClassifier(nn.Module):
    def __init__(self, feature_layer_depth=1., use_adjoint_method=True):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=2),
            nn.Conv2d(64, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # Downsamples an input tensor with shape (batch_size, 1, 28, 28)  
           # to a tensor with shape (batch_size, 64, 6, 6)

        if not isinstance(feature_layer_depth, float):
            raise TypeError('feature_layer_depth must be a floating-point number')
        if not isinstance(use_adjoint_method, bool):
            raise TypeError('use_adjoint_method must be True or False')
        self.feature_layer_depth = feature_layer_depth
        self.use_adjoint_method = use_adjoint_method
        self.ode_func = ODEFunction()
        self.ode_net = ODENetwork(self.ode_func)

        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        
    def forward(self, z):
        z = self.downsampling(z)
        t = torch.tensor([0., self.feature_layer_depth]).to(z)
        
        if self.use_adjoint_method: 
            z = self.ode_net(z, t)[-1]
        else:
            z = ode_solver(z, self.ode_func, t[0], t[1])
 
        z = self.bn(z)
        z = self.pool(z)
        z = z.reshape(z.shape[0], -1)
        z = self.fc(z)
        return z


def train_model():
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_train_loss = running_loss / len(train_loader)
    print(f'Training loss: {average_train_loss:.5f}')


def validate_model():
    model.eval()
    running_val_loss = 0
    correct_pred = 0
    global best_accuracy
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()

            predicted_labels = outputs.argmax(dim=1)
            correct_pred += (predicted_labels == labels).sum().item()

        average_val_loss = running_val_loss / len(val_loader)                  
        accuracy = 100. * correct_pred / len(val_loader.dataset) 
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'model.pt')
            best_accuracy = accuracy
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5 


def test_model():
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    correct_pred = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_labels = outputs.argmax(dim=1)
            correct_pred += (predicted_labels == labels).sum().item()
            
        accuracy = 100. * correct_pred / len(test_loader.dataset)    
    print(f'Test Accuracy: {accuracy:.3f}%')
    record_data(data_name='accuracy', header_row=['Accuracy'], data_row=[accuracy])


def measure_memory_and_time(func):
    gc.collect()
    if device.type == 'cuda':
        cuda.empty_cache()
        cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Peak memory: {cuda.max_memory_allocated() / 1024**2:.3f}MiB')
        print(f'Total time: {total_time:.2f}sec')
        record_data(
            data_name='memory', 
            header_row=[
            'Feature layer depth', 'Peak memory allocated'
            ], 
            data_row=[
            model.feature_layer_depth, cuda.max_memory_allocated()/ 1024**2, 
            ]
            )
        record_data(
            data_name='time', header_row=['Feature layer depth', 'Time'], 
            data_row=[model.feature_layer_depth, total_time]
            )


def record_data(data_name, header_row, data_row):
    file_name = data_name + ' (adjoint method).csv' \
        if model.use_adjoint_method else data_name + ' (standard backprop).csv'
    if not os.path.isfile(file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_row)
    with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'{num:.3f}' for num in data_row])
    

def get_data(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        data = [row for row in reader]
    return np.array(data, dtype=float)


if __name__ == '__main__':
    
    # Training and testing
    # To skip this step, set num_folds = 2 and num_epochs = 1. This will initialize
    # the train loader and allow the device to warm up for memory and time testing
    num_folds = 2 
    num_epochs = 1
    k_fold = KFold(num_folds, shuffle=True, random_state=10)

    if device.type == 'cpu':
        print("Currently running on CPU, which may take longer than GPU")

    for fold, (train_indices, val_indices) in enumerate(k_fold.split(dataset)):
        train_dataset, val_dataset =  Subset(dataset, train_indices), Subset(dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for flag in [True, False]:
            model = MNISTClassifier(use_adjoint_method=flag).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.004)
            best_accuracy = 0

            for itr in range(num_epochs):
                print(f'Training: Fold {fold + 1} | Model {2 - flag} | Epoch {itr + 1}')
                train_model()
                validate_model()

            test_model()

    # Measure memory cost and execution time
    for flag in [True, False]:
        for i in range(2):
            depth = 1. * i
            print(f'Training model {2 - flag} with depth {depth:.1f}')
            model = MNISTClassifier(feature_layer_depth=depth, use_adjoint_method=flag).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            measure_memory_and_time(train_model)

    # Plot memory cost
    data_adj = get_data('memory (adjoint method).csv')
    data_std = get_data('memory (standard backprop).csv')  
    plt.plot(data_adj[:, 0], data_adj[:, 1], label='Adjoint method')
    plt.plot(data_std[:, 0], data_std[:, 1], label='Standard backprop.')
    plt.xlabel(r'Depth $L$')
    plt.ylabel('Momory (MiB)')
    plt.legend(loc='best')
    plt.savefig('memory plot.pdf')
    plt.show()
        
    # Plot relative time
    data_adj = get_data('time (adjoint method).csv')
    data_std = get_data('time (standard backprop).csv') 
    plt.plot(data_adj[:, 0], data_adj[:, 1] / data_adj[-1, 1], label='Adjoint method')
    plt.plot(data_std[:, 0], data_std[:, 1] / data_adj[-1, 1], label='Standard backprop.')
    plt.xlabel(r'Depth $L$')
    plt.ylabel('Relative time')
    plt.legend(loc='best')
    plt.savefig('time plot.pdf')
    plt.show()    
