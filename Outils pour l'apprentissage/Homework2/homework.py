import torch
import torch.nn as nn
import torch.distributed as dist

"""
The goal of this homework is to implement a pipelined training loop.
Each rank holds one part of the model. It receives inputs from the previous rank and sends outputs to the next rank.

GPipe schedule:
--------------------------------------
Rank 0 | F F F F             B B B B |
Rank 1 |   F F F F         B B B B   |
Rank 2 |     F F F F     B B B B     |
Rank 3 |       F F F F B B B B       |
--------------------------------------

Command to run this file:
torchrun --nproc-per-node 4 homework.py
"""


def sequential_forward(model_part, inputs):
    """
    Handles the forward pass in a distributed pipeline
    
    - For all ranks except the first (rank 0), receives inputs from the previous rank
    - Processes the inputs through the local model segment
    - For all ranks except the last, sends the outputs to the next rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank != 0:
        # Receive inputs from the previous rank
        inputs = torch.zeros_like(inputs, requires_grad=True)
        dist.recv(inputs, src=rank - 1)
    
    # Process the inputs through the local model segment
    outputs = model_part(inputs)
    outputs.retain_grad()  # Retain gradients for non-leaf tensors
    
    if rank != world_size - 1:
        # Send outputs to the next rank
        dist.send(outputs, dst=rank + 1)

    return inputs, outputs  # both are needed for backward pass

def sequential_backward(inputs, outputs, targets, loss_fn):
    """
    Executes a backward pass in a pipeline-parallel distributed setup
    
    - Last rank computes the loss and backwards from there
    - Other ranks receive gradients from the next rank and perform backward on outputs with received gradients
    - All ranks except first send gradients to the previous rank

    hint: tensor.backward() can take a gradient tensor as an argument
    
    Returns the loss on the last rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == world_size - 1:
        # Last rank computes the loss
        loss = loss_fn(outputs, targets)
        loss.backward()  # Compute gradients
        grad_outputs = outputs.grad  # Extract the gradient of outputs for sending to the previous rank
    else:
        # Other ranks receive gradients from the next rank
        grad_outputs = torch.zeros_like(outputs)
        dist.recv(grad_outputs, src=rank + 1)  # Receive gradients from the next rank        
        outputs.backward(grad_outputs)

    if rank != 0:
        # Send gradients to the previous rank
        dist.send(inputs.grad, dst=rank - 1)

    if rank == world_size - 1:
        return loss

def pipelined_iteration(model_part, inputs, targets, loss_fn):
    """
    Implement one iteration of pipelined training using GPipe
    - Split the inputs and targets into microbatches
    - Perform forward passes for all microbatches (use sequential_forward)
    - Perform backward passes for all microbatches (use sequential_backward)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    microbatches = torch.chunk(inputs, world_size)
    microtargets = torch.chunk(targets, world_size)
    
    forward_cache = []
    total_loss = 0
    
    # Forward pass for all microbatches
    for i, microbatch in enumerate(microbatches):
        in_cache, out_cache = sequential_forward(model_part, microbatch)
        forward_cache.append((in_cache, out_cache))
    
    # Backward pass for all microbatches in reverse order
    for i, microtarget in reversed(list(enumerate(microtargets))):
        inputs, outputs = forward_cache[i]
        loss = sequential_backward(inputs, outputs, microtarget, loss_fn)
        if rank == world_size - 1:
            total_loss += loss.item()
    
    return total_loss


class MyDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for testing
    """
    def __init__(self, n = 1024):
        self.data = torch.randn(n, 32)
        self.targets = (self.data * 1.3) - 0.65
        # Synchronize data across all ranks
        with torch.no_grad():
            dist.broadcast(self.data, src = 0)
            dist.broadcast(self.targets, src = 0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def pipelined_training(model_part):
    """
    Perform pipelined training on a full dataset
    For each batch:
    - Perform pipelined iteration (use pipelined_iteration)
    - Update the model parameters
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_part.parameters())
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    for epoch in range(10):
        epoch_loss = 0
        # Iterate over the dataset in batches
        for batch in dataloader:
            inputs, targets = batch
            # Perform pipelined iteration
            total_loss = pipelined_iteration(model_part, inputs, targets, loss_fn)
            
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate loss
            if rank == world_size - 1:
                epoch_loss += total_loss
        
        if rank == world_size - 1:
            print(f"[Rank {rank}] Epoch {epoch} loss: {epoch_loss / len(dataset)}")


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # This is the full model
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Identity() # an even number of layers is easier to split
    )

    # Each rank gets a part of the model
    layers_per_rank = len(model) // world_size
    local_model = model[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} model: {local_model}")

    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets

    try:
        inputs, outputs = sequential_forward(local_model, inputs)
        print(f"[Rank {rank}] Sequential forward succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Sequential forward failed with error: {e}")

    try:
        sequential_backward(inputs, outputs, targets, nn.functional.mse_loss)
        print(f"[Rank {rank}] Sequential backward succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Sequential backward failed with error: {e}")

    try:
        pipelined_iteration(local_model, inputs, targets, nn.functional.mse_loss)
        print(f"[Rank {rank}] Pipeline iteration succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Pipeline iteration failed with error: {e}")

    try:
        pipelined_training(local_model)
        print(f"[Rank {rank}] Pipeline training succeeded")
    except Exception as e:
        print(f"[Rank {rank}] Pipeline training failed with error: {e}")

    dist.destroy_process_group()


"""
Additional question (optional):

Megatron-LM (https://arxiv.org/pdf/2104.04473) proposes a mechanism called "interleaving" (Section 2.2). Its idea is to assign multiple stages to each rank, instead of one.
- What is the main benefit of this approach?
- What is the main drawback?
- What would you change in the code to implement this?
"""