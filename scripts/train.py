import os
import argparse
import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from model import AItA_Classifier
from datasets import load_dataset
from dataset import TokenizedDataset
from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm as show_progress



def main(args):

    model = AItA_Classifier()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    start_epoch = 0
    data_seed = args.seed
    history = {
        'Train Loss': [],
        'Train Accuracy': [],
        'Validation Loss': [],
        'Validation Accuracy': []
    }
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        data_seed = checkpoint["seed"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    full_dataset = load_dataset('awhall/aita_21-11_22-10')

    label_mapping = { "Not the A-hole": 0, "Asshole": 1, "No A-holes here": 2, "Everyone Sucks": 3, "Not enough info": 4 }
    ds = full_dataset.map(lambda x: { **x, 'num_label': label_mapping[x['label']] }).class_encode_column('num_label')

    first_split = ds['train'].train_test_split(test_size=0.2, stratify_by_column="num_label", seed=data_seed)

    second_split = first_split['test'].train_test_split(test_size=0.5, stratify_by_column="num_label", seed=data_seed)

    batch_args = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True
    }

    train_batch_loader = DataLoader(TokenizedDataset(first_split['train']), **batch_args)

    validation_batch_loader = DataLoader(TokenizedDataset(second_split['train']), **batch_args)

    test_batch_loader = DataLoader(TokenizedDataset(second_split['test']), **batch_args)


    model.to(args.device)

    train_accuracy = Accuracy(task="multiclass", num_classes=5).to(args.device)
    validation_accuracy = Accuracy(task="multiclass", num_classes=5).to(args.device)

    for epoch in range(start_epoch, start_epoch + args.epochs):

        model.train()
        train_accuracy.reset()
        train_loss = 0

        print("Training...")
        for batch in show_progress(train_batch_loader):
            batch = [feature.to(args.device) for feature in batch]
            optimizer.zero_grad()
            logits = model(*batch[:-1])
            loss = criterion(logits, batch[-1])
            loss.backward()
            optimizer.step()
            train_accuracy(torch.max(logits, dim=1)[1], torch.max(batch[-1], dim=1)[1])
            train_loss += loss.item()

        model.eval()
        validation_accuracy.reset()
        validation_loss = 0
        print("Validation...")
        with torch.no_grad():
            for batch in show_progress(validation_batch_loader):
                batch = [feature.to(args.device) for feature in batch]
                logits = model(*batch[:-1])
                loss = criterion(logits, batch[-1])
                validation_accuracy(torch.max(logits, dim=1)[1], torch.max(batch[-1], dim=1)[1])
                validation_loss += loss.item()

        history['Train Loss'].append(train_loss/len(train_batch_loader))
        history['Train Accuracy'].append(train_accuracy.compute())
        history['Validation Loss'].append(validation_loss/len(validation_batch_loader))
        history['Validation Accuracy'].append(validation_accuracy.compute())

        metrics_str = '\n'.join(k +': '+f"{v[-1]:.4f}" for k, v in history.items())
        print(f"Epoch {epoch+1}:\n{metrics_str}")

        checkpoint = {
            'epoch': epoch,
            'seed': data_seed,
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(checkpoint, args.model_path)

    test_accuracy = Accuracy(task="multiclass", num_classes=5).to(args.device)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in show_progress(test_batch_loader):
            batch = [feature.to(args.device) for feature in batch]
            logits = model(*batch[:-1])
            loss = criterion(logits, batch[-1])
            test_accuracy(torch.max(logits, dim=1)[1], torch.max(batch[-1], dim=1)[1])
            test_loss += loss.item()

    test_loss /= len(test_batch_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy.compute():.4f}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="AItAS Verdict Prediction Training Script"
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="data/checkpoints/",
        help="Path to the directory where data related to training will be saved and loaded; model checkpoints, metrics, etc.",
        metavar=""
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="How many datapoints to include in each batch, increasing the batch_size may help speed up each training epoch, but using smaller batch_size may help with generalization because each batch is more noisy.",
        metavar=""
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help="AKA step-length, determines how much the optimizer adjusts the model weights in each step during gradient descent.",
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
        '--model_name',
        type=str,
        default="default",
        help="Provide a name to save and load model checkpoints under.",
        metavar=""
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Keeps the train test split consistent between runs. If resuming training this argument is loaded from checkpoint and will be ignored to maintain integrity.",
        metavar=""
    )

    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)

    args.model_path = f"{args.data_path}{args.model_name}-checkpoint.pth"

    args.device = 'cuda' if cuda.is_available() else 'cpu'

    main(args)