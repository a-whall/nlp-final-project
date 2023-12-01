import os
import argparse
import torch
from torch import (
    argmax,
    cuda,
    load,
    nn,
    no_grad,
    optim,
    save,
    Tensor,
    tensor
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchmetrics.classification import MulticlassAccuracy
from progress import ShowProgress
from inspect import (
    getmembers,
    isclass
)
import numpy as np
import models
from util import (
    plot,
    subsample,
    stratified_kfold,
    stratified_split
)



def main(args):

    ModelClass = getattr(models, args.model)

    model = ModelClass(args.dropout)
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
        checkpoint = load(args.model_path)
        args.seed = checkpoint["seed"]
        history = checkpoint["history"]
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(args.device)

    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False

    full_dataset = load_dataset('awhall/aita_21-10_23-09')["train"]
    keep = { "Not the A-hole": 15000, "Asshole": 15000, "No A-holes here": 9852, "Everyone Sucks": 7910, "Not enough info": 4552 }
    subsampled_dataset, weights = subsample(full_dataset, keep, args.seed)
    split = stratified_split(subsampled_dataset, "80/10/10", column="label", seed=args.seed)

    criterion = nn.CrossEntropyLoss(reduction='sum', weight=tensor(weights, dtype=torch.float, device=args.device))

    cpu_count = os.cpu_count()
    loader_args = {
        'batch_size': args.device_batch_size,
        'num_workers': cpu_count if cpu_count is not None else 1,
        'pin_memory': True,
        'shuffle': True
    }

    train_dataset = model.dataLoaderType(split['train'])
    train_batch_loader = DataLoader(train_dataset, **loader_args)

    validation_dataset = model.dataLoaderType(split['test'])
    validation_batch_loader = DataLoader(validation_dataset, **loader_args)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        if epoch == 0:
            loss, acc = eval(model, criterion, validation_batch_loader, args)    
            history["Validation Loss"].append(loss/len(validation_dataset))
            history["Validation Accuracy"].append(acc)
            history["Train Loss (epoch)"].append(None)
            history["Train Accuracy"].append(None)

        loss, batch_hist, acc = train(model, criterion, optimizer, train_batch_loader, args)
        history["Train Loss (epoch)"].append(loss/len(train_dataset))
        history["Train Loss (batch)"].extend(batch_hist)
        history["Train Accuracy"].append(acc)

        loss, acc = eval(model, criterion, validation_batch_loader, args)
        history["Validation Loss"].append(loss/len(validation_dataset))
        history["Validation Accuracy"].append(acc)

        plot("Batch Loss", "Steps", "Cross Entropy Loss", args,
            (history["Train Loss (batch)"], f"Batch Size: {args.batch_size}")
        )

        plot("Epoch Loss", "Epoch", "Cross Entropy Loss", args,
            (history["Train Loss (epoch)"], "Train"),
            (history["Validation Loss"], "Validation")
        )

        plot("Epoch Accuracy", "Epoch", "Macro Accuracy", args,
            (history["Train Accuracy"], "Train"),
            (history["Validation Accuracy"], "Validation")
        )

        checkpoint = {
            'epoch': epoch,
            'seed': args.seed,
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        save(checkpoint, args.model_path)



def train(model, criterion, optimizer, batch_loader, args):
    model.train()
    accuracy = MulticlassAccuracy(num_classes=5, average="macro").to(args.device)
    accumulated_loss = 0.0
    samples_processed = 0
    total_loss = 0.0
    batch_loss_history = []
    for i, batch in enumerate(ShowProgress(batch_loader, desc=f"Training")):
        batch = [feature.to(args.device) for feature in batch]            
        logits = model(*batch[:-1])
        targets = argmax(batch[-1], dim=1)
        loss = criterion(logits, targets)
        loss.backward()
        accuracy(argmax(logits, dim=1), targets)
        accumulated_loss += loss.item()
        samples_processed += batch[-1].size(0)
        if (i+1) % args.gradient_accumulation_steps == 0 or (i+1) == len(batch_loader):
            optimizer.step()
            optimizer.zero_grad()
            batch_loss_history.append(accumulated_loss/samples_processed)
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            samples_processed = 0
    return (total_loss, batch_loss_history, accuracy.compute().cpu().item())



def eval(model: nn.Module, criterion: nn.CrossEntropyLoss, batch_loader: DataLoader, args: dict):
    """
    Purpose:
        Set the model to eval mode and evaluate its performance over an entire dataset.
    """
    model.eval()
    accuracy = MulticlassAccuracy(num_classes=5, average="macro").to(args.device)
    loss = 0.0
    with no_grad():
        for batch in ShowProgress(batch_loader, desc=f"Evaluating"):
            batch = [feature.to(args.device) for feature in batch]
            targets = argmax(batch[-1], dim=1)
            logits = model(*batch[:-1])
            accuracy(argmax(logits, dim=1), targets)
            loss += criterion(logits, targets).item()
    return (loss, accuracy.compute().cpu().item())



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AItAS Verdict Prediction Model Training Script"
    )

    parser.add_argument(
        '--model',
        type=str,
        default="DistilBert_A",
        choices=[name for name, obj in getmembers(models) if isclass(obj) and issubclass(obj, nn.Module)],
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
        '--weight_decay',
        type=float,
        default=1e-5,
        help="L2 regularization factor.",
        metavar=""
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help="Model dropout rate. Percent chance of an element in the dropout layer to be zeroed.",
        metavar=""
    )

    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=16,
        help="How many batches will be processed before allowing the optimizer to update model weights. Larger batch size often result in smoother gradients which can lead to more stable and reliable training convergence, smaller batch size tend to produce noisy gradients which may help the model to generalize better. When adjusting batch size, consider adjusting the learning rate as well.",
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

    args.batch_size = args.device_batch_size * args.gradient_accumulation_steps

    args.model_name = f"{args.model}-{args.seed}-{args.batch_size}-{args.freeze}-{args.dropout*100:.0f}-{args.weight_decay:.0e}"

    args.model_path = f"{args.data_path}{args.checkpoint_dir}{args.model_name}.pth"

    args.start_epoch = 0

    main(args)