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
import matplotlib.pyplot as plt

import loralib as lora
from torch.optim.lr_scheduler import LambdaLR

# Initialize wandb
wandb.init(project="simple-neural-net-3", settings=wandb.Settings(start_method="thread"))

seed = 42
torch.manual_seed(seed)
random.seed(seed)

# Define a simple 2-layer neural network
class SimpleNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, rank, add_lora=False):
        super(SimpleNet, self).__init__()
        linear = lora.Linear if add_lora else nn.Linear
        embedding = lora.Embedding if add_lora else nn.Embedding
        self.relu = nn.ReLU()

        if add_lora:
            self.embedding = embedding(vocab_size, hidden_size, r=rank)
            self.fc1 = linear(hidden_size, hidden_size, r=rank)
            self.fc2 = linear(hidden_size, output_size, r=rank)
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
    rank = args.rank
    add_lora = args.add_lora

    # Load and preprocess dataset
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    model = SimpleNet(vocab_size, hidden_size, output_size, rank, add_lora)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Total number of training steps
    total_steps = num_epochs * len(train_loader)
    # Linear learning rate decay function
    lr_lambda = lambda step: max(0.0, 1.0 - step / total_steps)
    # Learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


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

    # Initialize empty lists to store heatmaps
    heatmap_list_emb = []
    heatmap_list_fc1 = []
    heatmap_list_fc2 = []


    if add_lora:
        Wo_emb = ((model.embedding.lora_A).T @ (model.embedding.lora_B).T * 1/16).clone()
        Wo_fc1 = ((model.fc1.lora_A).T @ (model.fc1.lora_B).T * 1/16).clone()
        Wo_fc2 = ((model.fc2.lora_A).T @ (model.fc2.lora_B).T * 1/16).clone()

        heatmap_list_emb.append(Wo_emb.detach().cpu().numpy())
        heatmap_list_fc1.append(Wo_fc1.detach().cpu().numpy())
        heatmap_list_fc2.append(Wo_fc2.detach().cpu().numpy())
    else:
        Wo_emb = model.embedding.weight.clone()
        Wo_fc1 = model.fc1.weight.clone()
        Wo_fc2 = model.fc2.weight.clone()

        heatmap_list_emb.append(Wo_emb.detach().cpu().numpy())
        heatmap_list_fc1.append(Wo_fc1.detach().cpu().numpy())
        heatmap_list_fc2.append(Wo_fc2.detach().cpu().numpy())

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            labels = torch.LongTensor(batch['label']).squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
           
            if add_lora:
                optimizer.step()
                wandb.log({"update_WaWb_embed": ((model.embedding.lora_A).T @ (model.embedding.lora_B).T * 1/rank).norm().item()})
                wandb.log({"update_WaWb_fc1": ((model.fc1.lora_A).T @ (model.fc1.lora_B).T * 1/rank).norm().item()})
                wandb.log({"update_WaWb_fc2": ((model.fc2.lora_A).T @ (model.fc2.lora_B).T * 1/rank).norm().item()})
            else:
                optimizer.step()
                wandb.log({"update_W_embed": ((model.embedding.lora_A).T @ (model.embedding.lora_B).T * 1/rank).norm().item()})
                wandb.log({"update_W_fc1": ((model.fc1.lora_A).T @ (model.fc1.lora_B).T * 1/rank).norm().item()})
                wandb.log({"update_W_fc2": ((model.fc2.lora_A).T @ (model.fc2.lora_B).T * 1/rank).norm().item()})

        # after each epoch
        if add_lora:       
            print("enter the loop")
            W_emb_current = ((model.embedding.lora_A).T @ (model.embedding.lora_B).T * 1/rank).clone().detach()
            W_fc1_current = ((model.fc1.lora_A).T @ (model.fc1.lora_B).T * 1/rank).clone().detach()
            W_fc2_current = ((model.fc2.lora_A).T @ (model.fc2.lora_B).T * 1/rank).clone().detach()

            delta_W_emb = W_emb_current - W0_emb
            delta_W_fc1 = W_fc1_current - W0_fc1
            delta_W_fc2 = W_fc2_current - W0_fc2

            wandb.log({"Span_update_WaWb_embed": delta_W_emb.norm().item()})
            wandb.log({"Span_update_WaWb_fc1": delta_W_fc1.norm().item()})
            wandb.log({"Span_update_WaWb_fc2": delta_W_fc2.norm().item()})

            # Append the heatmaps to the respective lists
            heatmap_list_emb.append(delta_W_emb.detach().cpu().numpy())
            heatmap_list_fc1.append(delta_W_fc1.detach().cpu().numpy())
            heatmap_list_fc2.append(delta_W_fc2.detach().cpu().numpy())

            # Update W0 for the next 1000 steps
            W0_emb = W_emb_current
            W0_fc1 = W_fc1_current
            W0_fc2 = W_fc2_current
                
            # # Calculate the norms of Wa and Wb
            # norm_Wa = torch.norm(model.embedding.lora_A)
            # norm_Wb = torch.norm(model.embedding.lora_B)
            # wandb.log({"norm_Wa_embed": norm_Wa.item()})
            # wandb.log({"norm_Wb_embed": norm_Wb.item()})
            # update_Wa = optimizer.param_groups[0]['lr'] * optimizer.state[model.embedding.lora_A]['exp_avg'] / (torch.sqrt(optimizer.state[model.embedding.lora_A]['exp_avg_sq']) + 1e-8)
            # update_Wb = optimizer.param_groups[0]['lr'] * optimizer.state[model.embedding.lora_B]['exp_avg'] / (torch.sqrt(optimizer.state[model.embedding.lora_B]['exp_avg_sq']) + 1e-8)
            # wandb.log({"update_Wa_embed": update_Wa.norm().item()})
            # wandb.log({"update_Wb_embed": update_Wb.norm().item()})

            # # Log norms and updates for fc1 layer
            # wandb.log({"WaWb_fc1_layer": ((model.fc1.lora_A).T @ (model.fc1.lora_B).T * 1/16).norm().item()})
            # norm_Wa_fc1 = torch.norm(model.fc1.lora_A)
            # norm_Wb_fc1 = torch.norm(model.fc1.lora_B)
            # wandb.log({"norm_Wa_fc1": norm_Wa_fc1.item()})
            # wandb.log({"norm_Wb_fc1": norm_Wb_fc1.item()})
            # update_Wa_fc1 = optimizer.param_groups[0]['lr'] * optimizer.state[model.fc1.lora_A]['exp_avg'] / (torch.sqrt(optimizer.state[model.fc1.lora_A]['exp_avg_sq']) + 1e-8)
            # update_Wb_fc1 = optimizer.param_groups[0]['lr'] * optimizer.state[model.fc1.lora_B]['exp_avg'] / (torch.sqrt(optimizer.state[model.fc1.lora_B]['exp_avg_sq']) + 1e-8)
            # wandb.log({"update_Wa_fc1": update_Wa_fc1.norm().item()})
            # wandb.log({"update_Wb_fc1": update_Wb_fc1.norm().item()})

            # # Log norms and updates for fc2 layer
            # wandb.log({"WaWb_fc2_layer": ((model.fc2.lora_A).T @ (model.fc2.lora_B).T * 1/16).norm().item()})
            # norm_Wa_fc2 = torch.norm(model.fc2.lora_A)
            # norm_Wb_fc2 = torch.norm(model.fc2.lora_B)
            # wandb.log({"norm_Wa_fc2": norm_Wa_fc2.item()})
            # wandb.log({"norm_Wb_fc2": norm_Wb_fc2.item()})
            # update_Wa_fc2 = optimizer.param_groups[0]['lr'] * optimizer.state[model.fc2.lora_A]['exp_avg'] / (torch.sqrt(optimizer.state[model.fc2.lora_A]['exp_avg_sq']) + 1e-8)
            # update_Wb_fc2 = optimizer.param_groups[0]['lr'] * optimizer.state[model.fc2.lora_B]['exp_avg'] / (torch.sqrt(optimizer.state[model.fc2.lora_B]['exp_avg_sq']) + 1e-8)
            # wandb.log({"update_Wa_fc2": update_Wa_fc2.norm().item()})
            # wandb.log({"update_Wb_fc2": update_Wb_fc2.norm().item()})

        else:
            print("enter the loop")
            updated_weights_embed = model.embedding.weight.clone()
            diff_emb = updated_weights_embed - Wo_emb
            heatmap_list_emb.append(diff_emb.detach().cpu().numpy())
            wandb.log({"Span_update_W_embed": diff_emb.norm().item()})

            # Track the weight updates for fc1 layer
            updated_weights_fc1 = model.fc1.weight.clone()
            diff_fc1 = updated_weights_fc1 - W0_fc1
            heatmap_list_fc1.append(diff_fc1.detach().cpu().numpy())
            wandb.log({"Span_update_W_fc1": diff_fc1.norm().item()})

            # Track the weight updates for fc2 layer
            updated_weights_fc2 = model.fc2.weight.clone()
            diff_fc2 = updated_weights_fc2 - W0_fc2
            heatmap_list_fc2.append(diff_fc2.detach().cpu().numpy())
            wandb.log({"Span_update_W_fc2": diff_fc1.norm().item()})

            # Update W0 for the next 1000 steps
            W0_emb = updated_weights_embed
            W0_fc1 = updated_weights_fc1
            W0_fc2 = updated_weights_fc2

            # initial_weights_embed = model.embedding.weight.clone()
            # optimizer.step()
            # updated_weights_embed = model.embedding.weight.clone()
            # diff = (updated_weights_embed-initial_weights_embed).norm().item()
            # wandb.log({"norm(updated-initial)": diff})

            # # Accessing the embedding matrix and its update in Adam optimizer
            # first_moment = optimizer.state[model.embedding.weight]['exp_avg']  # Gradient (m1)
            # second_moment = optimizer.state[model.embedding.weight]['exp_avg_sq']  # Update (m2)
            # # Calculate the update without modifying the original parameters
            # embedding_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
            # # Logging the update without applying it to the parameters
            # wandb.log({"Adam(embedding_grad)": embedding_update.norm().item()})

            # first_moment = optimizer.state[model.fc1.weight]['exp_avg']  # Gradient (m1)
            # second_moment = optimizer.state[model.fc1.weight]['exp_avg_sq']  # Update (m2)
            # fc1_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
            # wandb.log({"Adam(fc1_grad)": fc1_update.norm().item()})

            # first_moment = optimizer.state[model.fc2.weight]['exp_avg']  # Gradient (m1)
            # second_moment = optimizer.state[model.fc2.weight]['exp_avg_sq']  # Update (m2)
            # fc2_update = optimizer.param_groups[0]['lr'] * first_moment / (torch.sqrt(second_moment) + 1e-8)
            # wandb.log({"Adam(fc2_grad)": fc2_update.norm().item()})
            # # wandb.log({"embedding_grad": model.embedding.weight.grad.norm().item(), "fc1_grad": model.fc1.weight.grad.norm().item(), "fc2_grad": model.fc2.weight.grad.norm().item()})
            
            scheduler.step()
            # Log loss and gradients to wandb after each epoch
            wandb.log({"train_loss": loss.item()})
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr})
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

    # After training, visualize and save all the heatmaps
    plt.figure(figsize=(18, 6))

    # Subplot for the embedding layer
    plt.subplot(1, 3, 1)
    for i, heatmap in enumerate(heatmap_list_emb):
        plt.imshow(heatmap, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Step {500 * (i + 1)}')
        plt.xlabel('Output Dimension')
        plt.ylabel('Input Dimension')

    # Subplot for the fc1 layer
    plt.subplot(1, 3, 2)
    for i, heatmap in enumerate(heatmap_list_fc1):
        plt.imshow(heatmap, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Step {500 * (i + 1)}')
        plt.xlabel('Output Dimension')
        plt.ylabel('Input Dimension')

    # Subplot for the fc2 layer
    plt.subplot(1, 3, 3)
    for i, heatmap in enumerate(heatmap_list_fc2):
        plt.imshow(heatmap, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Step {500 * (i + 1)}')
        plt.xlabel('Output Dimension')
        plt.ylabel('Input Dimension')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'all_heatmaps_combined_{add_lora}_{learning_rate}_{rank}.png')

    # Show the figure (optional)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of the hidden layer')
    parser.add_argument('--output_size', type=int, default=2, help='Size of the output layer')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--rank', type=int, default=16, help='Number of training epochs')
    parser.add_argument('--add_lora', action='store_true', help='Whether to add lora')


    args = parser.parse_args()
    main(args)