from datasets import load_dataset

dataset = load_dataset('awhall/aita_21-11_22-10', data_files={'all_data': 'AItAS_dataset.csv'})

split_datasets = dataset['all_data'].train_test_split(test_size=0.2)

val_test_datasets = split_datasets['test'].train_test_split(test_size=0.5)

train_dataset = split_datasets['train']
validation_dataset = val_test_datasets['train']
test_dataset = val_test_datasets['test']

def valid_label(text):
    return text == "Not the A-hole" or text == "Asshole" or text == "No A-holes here" or text == "Everyone Sucks" or text == "Not enough info"

for sample in train_dataset:
    if not valid_label(sample["label"]):
        print(sample["label"])