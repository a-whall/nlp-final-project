import os
import argparse
import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
import models
from datasets import load_dataset
from torchmetrics.classification import MulticlassAccuracy
from progress import ShowProgress
import matplotlib.pyplot as plt
from inspect import getmembers, isclass
import numpy as np



def main(args):

    ModelClass = getattr(models, args.model)

    model = ModelClass()
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    history = {
        'Train Loss (batch)': [],
        'Train Loss (epoch)': [],
        'Train Accuracy': [],
        'Validation Loss': [],
        'Validation Accuracy': []
    }
    if os.path.exists(args.model_path):
        print(f"Using model checkpoint {args.model_path}")
        checkpoint = torch.load(args.model_path)
        args.seed = checkpoint["seed"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    label_mapping = { "Not the A-hole": 0, "Asshole": 1, "No A-holes here": 2, "Everyone Sucks": 3, "Not enough info": 4 }
    full_dataset = load_dataset('awhall/aita_21-10_23-09')["train"]
    ds = full_dataset.map(lambda x: { **x, 'num_label': label_mapping[x['label']] }).class_encode_column('num_label')
    
    # Class balance the dataset.
    np.random.seed(args.seed)
    N_to_keep = 4500
    for label, value in label_mapping.items():
        indices_of_label = [i for i, example in enumerate(ds) if example['num_label'] == value]
        indices_to_keep = np.random.choice(indices_of_label, N_to_keep, replace=False)
        indices_of_other_labels = [i for i, example in enumerate(ds) if example['num_label'] != value]
        indices_to_keep = np.concatenate((indices_of_other_labels, indices_to_keep))
        ds = ds.select(indices_to_keep)

    # Show class distribution.
    label_counts = {"Not the A-hole": 0, "Asshole": 0, "No A-holes here": 0, "Everyone Sucks": 0, "Not enough info": 0}
    for sample in ds:
        label_counts[sample["label"]] += 1
    print(label_counts)

    split = ds.train_test_split(test_size=0.2, stratify_by_column="num_label", seed=args.seed)

    loader_args = {
        'batch_size': args.device_batch_size,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True
    }

    train_dataset = model.dataLoaderType(split['train'])
    train_batch_loader = DataLoader(train_dataset, **loader_args)

    validation_dataset = model.dataLoaderType(split['test'])
    validation_batch_loader = DataLoader(validation_dataset, **loader_args)

    train_accuracy = MulticlassAccuracy(num_classes=5, average="macro").to(args.device)
    validation_accuracy = MulticlassAccuracy(num_classes=5, average="macro").to(args.device)

    for epoch in range(start_epoch, start_epoch + args.epochs):

        # Evaluate the untrained model.
        if epoch == 0:
            model.eval()
            history["Train Loss (epoch)"].append(None)
            history["Train Accuracy"].append(None)
            validation_accuracy.reset()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in ShowProgress(validation_batch_loader, desc=f"Untrained Model Evaluation"):
                    batch = [feature.to(args.device) for feature in batch]
                    targets = torch.argmax(batch[-1], dim=1)
                    logits = model(*batch[:-1])
                    loss = criterion(logits, targets)
                    validation_accuracy(torch.argmax(logits,dim=1), targets)
                    validation_loss += loss.item()
                validation_loss /= len(validation_dataset)
            history["Validation Loss"].append(validation_loss)
            history["Validation Accuracy"].append(validation_accuracy.compute().cpu())

        # Training Loop.
        model.train()
        train_accuracy.reset()
        accumulation_steps = args.batch_size // args.device_batch_size
        accumulated_loss = 0.0
        samples_processed = 0
        total_loss = 0.0
        for i, batch in enumerate(ShowProgress(train_batch_loader, desc=f"Training Epoch-{epoch+1}")):
            batch = [feature.to(args.device) for feature in batch]            
            logits = model(*batch[:-1])
            targets = torch.argmax(batch[-1], dim=1)
            loss = criterion(logits, targets)
            loss.backward()
            train_accuracy(torch.argmax(logits,dim=1), targets)
            accumulated_loss += loss.item()
            total_loss += loss.item()
            samples_processed += batch[-1].size(0)
            if (i+1) % accumulation_steps == 0 or (i+1) == len(train_batch_loader):
                optimizer.step()
                optimizer.zero_grad()
                # The average loss per sample over the most recent batch.
                history["Train Loss (batch)"].append(accumulated_loss/samples_processed)
                accumulated_loss = 0.0
                samples_processed = 0
        # The average loss per sample over the entire training loop
        history["Train Loss (epoch)"].append(total_loss/len(train_dataset))

        plt.plot(history["Train Loss (batch)"], label=f"Batch Size: {args.batch_size}, Total Epochs: {epoch+1}")
        plt.xlabel("Steps")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(f"{args.data_path}{args.plot_dir}BatchLoss-{args.model_name}.png")
        plt.clf()

        # Evaluate the trained model.
        model.eval()
        validation_accuracy.reset()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in ShowProgress(validation_batch_loader, desc=f"Validating Epoch-{epoch+1}"):
                batch = [feature.to(args.device) for feature in batch]
                targets = torch.argmax(batch[-1], dim=1)
                logits = model(*batch[:-1])
                loss = criterion(logits, targets)
                validation_accuracy(torch.argmax(logits,dim=1), targets)
                validation_loss += loss.item()
            validation_loss /= len(validation_dataset)
        # Average loss per sample over the entire validation loop
        history["Validation Loss"].append(validation_loss)
        history["Train Accuracy"].append(train_accuracy.compute().cpu())
        history["Validation Accuracy"].append(validation_accuracy.compute().cpu())

        plt.plot(history["Train Loss (epoch)"], label=f"Train Loss (Average per sample over entire epoch)")
        plt.plot(history["Validation Loss"], label=f"Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(f"{args.data_path}{args.plot_dir}Loss-{args.model_name}.png")
        plt.clf()

        plt.plot(history["Train Accuracy"], label=f"Train")
        plt.plot(history["Validation Accuracy"], label=f"Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(f"{args.data_path}{args.plot_dir}Accuracy-{args.model_name}.png")
        plt.clf()

        checkpoint = {
            'epoch': epoch,
            'seed': args.seed,
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(checkpoint, args.model_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AItAS Verdict Prediction Model Training Script"
    )

    parser.add_argument(
        '--model',
        type=str,
        default="DistilBert_A",
        choices=[name for name, obj in getmembers(models) if isclass(obj) and issubclass(obj, torch.nn.Module)],
        help="The name of the model class to train. The name must match one of the models defined in scripts/models.py",
        metavar=""
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="data/",
        help="Path to the directory where data related to training will be saved and loaded; model checkpoints, metrics, etc.",
        metavar=""
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default="checkpoints/",
        help="The directory to store model checkpoints.",
        metavar=""
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default="plots/",
        help="The directory to store training plots.",
        metavar=""
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help="AKA step-length. This hyperparameter controls how much the optimizer adjusts the model weights in each step during gradient descent.",
        metavar=""
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help="How many inputs to include in each batch of optimization. This many inputs will be processed and contribute to gradient of model parameters before allowing the optimizer to update model weights. Larger batch size often result in smoother gradients which can lead to more stable and reliable training convergence, smaller batch size tend to produce noisy gradients which may help the model to generalize better. When adjusting batch size, consider adjusting the learning rate as well.",
        metavar=""
    )

    parser.add_argument(
        '--device_batch_size',
        type=int,
        default=8,
        help="How many inputs to attempt to load into memory at once. Must be less than or equal to batch_size. The optimal number will require some testing.",
        metavar=""
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="The number of epochs to do on this run.",
        metavar=""
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Keeps the train test split consistent between runs. If resuming training this argument is loaded from checkpoint and will be ignored to maintain integrity.",
        metavar=""
    )

    parser.add_argument(
        '--freeze',
        type=bool,
        default=False,
        help="Freeze the weights of the models bert layers. Keeps embeddings consistent.",
        metavar=""
    )

    args = parser.parse_args()

    os.makedirs(f"{args.data_path}{args.checkpoint_dir}", exist_ok=True)
    os.makedirs(f"{args.data_path}{args.plot_dir}", exist_ok=True)

    args.device = 'cuda' if cuda.is_available() else 'cpu'

    args.model_name = f"model-{args.model}-{args.seed}-{args.batch_size}-{args.freeze}"

    args.model_path = f"{args.data_path}{args.checkpoint_dir}{args.model_name}.pth"

    main(args)