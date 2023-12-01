import numpy as np
import matplotlib.pyplot as plt
from progress import ShowProgress
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset



def stratified_kfold(dataset: Dataset, column: str, k: int=3, seed: int = None) -> list[list[int]]:
    """
    Purpose:
        Create k evenly sized splits. Stratify splits on string or number.
    Args:
        dataset: Dataset to fold.
        column: Name of the column to split on.
        k: Number of splits to make.
        seed: Set this for reproducible folds.
    Return:
        A list containing k lists of indices.
    """
    rng = np.random.default_rng(seed)
    P = len(dataset)
    labels = np.array(dataset[column])
    indices = np.arange(P)
    folds = [[] for _ in range(k)]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_ratios = label_counts / P
    label_indices = { l: indices[labels == l] for l in unique_labels }
    for idx, label in enumerate(unique_labels):
        rng.shuffle(label_indices[label])
        start = 0
        for i in range(k):
            num_samples_for_fold = int(np.floor(label_ratios[idx] * (P // k + (i < P % k))))
            end = start + num_samples_for_fold
            folds[i].extend(label_indices[label][start:end].astype(int))
            start = end
    return folds



def subsample(dataset: Dataset, keep_counts: dict, seed: int = None, return_inverse_frequency_weights: bool = True) -> Dataset:
    """
    Purpose:
        Downsample a Dataset to the desired per-class quantities.
    Args:
        dataset: Dataset to subsample.
        keep_counts: {class-label: downsample-quantity foreach class-label}.
        seed: Set this for reproducible subsamples.
        return_inverse_frequency_weights: If True, computes and returns the inverse-frequency of downsample-quantities for loss penalty.
    Return:
        A Dataset, with reduced class quantities, and optionally, the loss weights.
    Note:
        keep_counts must give a quantity smaller than that which exists in the original dataset or np.random.choice will raise an exception.
        Be careful to pass the order of keep_count keys according to the order of labels in the model.
    """
    rng = np.random.default_rng(seed)
    subsample_size = sum(keep_counts.values())
    indices = {label: [] for label in keep_counts.keys()}
    for i, sample in enumerate(dataset):
        indices[sample["label"]].append(i)
    indices_to_keep = []
    for label, count in keep_counts.items():
        indices_to_keep.extend(rng.choice(indices[label], count, replace=False))
    if return_inverse_frequency_weights:
        weights = np.array([subsample_size/n for n in keep_counts.values()])
        weights /= weights.min()
        return (dataset.select(indices_to_keep), weights)
    return dataset.select(indices_to_keep)



def stratified_split(dataset: Dataset, split: str, column: str, seed: int = None) -> DatasetDict:
    """
    Purpose:
        Generate 3 splits of a Dataset for "train", "dev", and "test".
    Args:
        dataset: the full dataset to split.
        split: A string of the form "80/10/10" for example. Must sum to 100.
        seed: Set this for reproducible splits.
    Return:
        A DatasetDict with keys "train", "dev", "test".
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset[column])
    proportions = np.array([float(x)/100 for x in split.split('/')])
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    indices = np.arange(len(dataset))
    label_indices = {label: indices[labels == label] for label in unique_labels}
    split_indices = [[], [], []]
    for label in unique_labels:
        rng.shuffle(label_indices[label])
        start = 0
        for i in range(3):
            num_samples_for_fold = int(np.floor(proportions[i] * label_counts[unique_labels == label]))
            end = start + num_samples_for_fold
            split_indices[i].extend(label_indices[label][start:end].astype(int))
            start = end
    return DatasetDict({
        "train": dataset.select(split_indices[0]), # try keep_in_memory to avoid polluting .cache
        "dev": dataset.select(split_indices[1]),
        "test": dataset.select(split_indices[2])
    })



def plot(title: str, xlabel: str, ylabel: str, args: dict, *history_label_pairs):
    for history, label in history_label_pairs:
        plt.plot(history, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{args.data_path}{args.plot_dir}{title.replace(' ','')}-{args.model_name}.png")
    plt.clf()