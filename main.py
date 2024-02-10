# https://github.com/microsoft/LoRA
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

import loralib as lora

# Initialize wandb
wandb.init(project="simple-neural-net", settings=wandb.Settings(start_method="thread"))

seed = 42
torch.manual_seed(seed)
random.seed(seed)

# Define a simple 2-layer neural network
class SimpleNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, add_lora=False):
        super(SimpleNet, self).__init__()
        linear = lora.Linear if add_lora else nn.Linear
        embedding = lora.Embedding if add_lora else nn.Embedding
        self.relu = nn.ReLU()

        if add_lora:
            self.embedding = embedding(vocab_size, hidden_size, r=16)
            self.fc1 = linear(hidden_size, hidden_size, r=16)
            self.fc2 = linear(hidden_size, output_size, r=16)
        else:
            self.embedding = embedding(vocab_size, hidden_size)
            self.fc1 = linear(hidden_size, hidden_size)
            self.fc2 = linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average pooling over the sequence dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main(args):
    # Load IMDb dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenization(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    train_data = dataset['train'].map(tokenization, batched=True)
    test_data = dataset['test'].map(tokenization, batched=True)

    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    add_lora = args.add_lora

    # Load and preprocess dataset
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    model = SimpleNet(vocab_size, hidden_size, output_size, add_lora)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Log hyperparameters to wandb
    wandb.config.update({
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "add_lora": add_lora,
    })

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize WandB watch for gradients
    wandb.watch(model, log="all")
    
    print("add_lora = ", add_lora)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            labels = torch.LongTensor(batch['label']).squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
           
            if add_lora:
                optimizer.step()
                print(optimizer)
                wandb.log({"WaWb_embedding_layer": ((model.embedding.lora_A).T @ (model.embedding.lora_B).T * 1/16).norm().item()})
                # wandb.log({"WaWb_fc1_layer": ((model.fc1.lora_A).T @ (model.fc1.lora_B).T).norm().item()})
                # wandb.log({"WaWb_fc2_layer": ((model.fc1.lora_A).T @ (model.fc1.lora_B).T).norm().item()})
                # Calculate the norms of Wa and Wb
                norm_Wa = torch.norm(model.embedding.lora_A)
                norm_Wb = torch.norm(model.embedding.lora_B)

                # Log the norms of Wa and Wb
                wandb.log({"norm_Wa": norm_Wa.item()})
                wandb.log({"norm_Wb": norm_Wb.item()})

                # Calculate the updates for Wa and Wb
                update_Wa = optimizer.param_groups[0]['lr'] * optimizer.state[model.embedding.lora_A]['exp_avg'] / (torch.sqrt(optimizer.state[model.embedding.lora_A]['exp_avg_sq']) + 1e-8)
                update_Wb = optimizer.param_groups[0]['lr'] * optimizer.state[model.embedding.lora_B]['exp_avg'] / (torch.sqrt(optimizer.state[model.embedding.lora_B]['exp_avg_sq']) + 1e-8)

                # Log the norms of updates for Wa and Wb
                wandb.log({"update_Wa": update_Wa.norm().item()})
                wandb.log({"update_Wb": update_Wb.norm().item()})

            else:
                optimizer.step()
                # Accessing the embedding matrix and its update in Adam optimizer
                first_moment = optimizer.state[model.fc1.weight]['exp_avg']  # Gradient (m1)
                second_moment = optimizer.state[model.fc1.weight]['exp_avg_sq']  # Update (m2)
                # Calculate the update without modifying the original parameters
                embedding_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
                # Logging the update without applying it to the parameters
                wandb.log({"Adam(embedding_grad)": embedding_update.norm().item()})

                # first_moment = optimizer.state[model.fc1.weight]['exp_avg']  # Gradient (m1)
                # second_moment = optimizer.state[model.fc1.weight]['exp_avg_sq']  # Update (m2)
                # fc1_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
                # wandb.log({"Adam(fc1_grad)": fc1_update.norm().item()})

                # first_moment = optimizer.state[model.fc2.weight]['exp_avg']  # Gradient (m1)
                # second_moment = optimizer.state[model.fc2.weight]['exp_avg_sq']  # Update (m2)
                # fc2_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
                # wandb.log({"Adam(fc2_grad)": fc2_update.norm().item()})
                # # wandb.log({"embedding_grad": model.embedding.weight.grad.norm().item(), "fc1_grad": model.fc1.weight.grad.norm().item(), "fc2_grad": model.fc2.weight.grad.norm().item()})
            
           
            # Log loss and gradients to wandb after each epoch
            wandb.log({"train_loss": loss.item()})
       
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            labels = torch.LongTensor(batch['label']).squeeze().to(device)

            outputs = model(input_ids)
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    wandb.log({"test_accuracy": accuracy})
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of the hidden layer')
    parser.add_argument('--output_size', type=int, default=2, help='Size of the output layer')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--add_lora', action='store_true', help='Whether to add lora')

    args = parser.parse_args()
    main(args)