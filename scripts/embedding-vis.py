"""
Purpose:
    An attempt at visualizing distances between bert embeddings of our AItA posts by class.
    Inspiration: https://www.kaggle.com/code/mateiionita/visualizing-bert-embeddings-with-t-sne/notebook
TODO:
    - optimize to use GPU, if available.
    - pass batched inputs to model.
    - use datasets lib to access the dataset instead of an expected local copy.
"""
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import csv
from transformers import DistilBertTokenizer, DistilBertModel



N_SAMPLES_PER_CLASS = 50
SEED = 0
DATAPATH = 'data/AItAS_dataset.csv'
CLASSES = ["Not the A-hole", "Asshole", "No A-holes here", "Everyone Sucks", "Not enough info"]



os.makedirs("./data/")

# Separate the data by class.
class_separated_data = { c: [] for c in CLASSES }
with open(DATAPATH, 'r', encoding='utf8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        if row:
            class_separated_data[row[2]].append(row[0:2])

# Subsample each category
subsampled_data = {}
for category in CLASSES:
    total_samples = len(class_separated_data[category])
    random_indices = np.random.choice(total_samples, N_SAMPLES_PER_CLASS, replace=False)
    subsampled_data[category] = [class_separated_data[category][i][1] for i in random_indices]

# Generate embeddings
embeddings = []
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
for category in CLASSES:
    for i in range(N_SAMPLES_PER_CLASS):
        text = subsampled_data[category][i]
        inputs = tokenizer(
            text,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding='max_length'
        )
        embedding = distilbert(
            torch.tensor(inputs["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(inputs["attention_mask"], dtype=torch.long),
            return_dict=False
        )[0][:,0,:][0]
        embeddings.append(embedding.detach().numpy())
embeddings_numpy = np.vstack(embeddings)

# Dimensionality reduction
pca_data = PCA(n_components=50).fit_transform(embeddings)
tsne_data = TSNE(n_components=2, random_state=SEED).fit_transform(pca_data)
print(tsne_data.shape)

# Produce the plot
plt.figure(figsize=(12,8))
for i, category in enumerate(CLASSES):
    indices = np.arange(i*N_SAMPLES_PER_CLASS, (i+1)*N_SAMPLES_PER_CLASS)
    plt.scatter(tsne_data[indices,0], tsne_data[indices,1], label=category)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE on the top 50 Principal Components of DistilBert Embeddings')
plt.legend()
plt.savefig('./data/t-sne-plot.png')