# r/AmItheAsshole Verdict Prediction Using BERT



# Setup

0. Clone the repository

1. Initialize a new virtual environment
- (On Unix or MacOS): `python3 -m venv venv`
- (On Windows): `py -m venv venv`

2. Activate the virtual environment
- (On Unix or MacOS): `source venv/bin/activate`
- (On Windows): `venv\Scripts\activate`

3. Install the necessary Python packages
- `pip install -r requirements.txt`

4. Install PyTorch
- The appropriate Torch library is dependent on system hardware, go [here](https://pytorch.org/get-started/locally/#start-locally) to get the command for your system, use CUDA if you have a capable device. Note: for this project we do not need the torchvision or torchaudio library extensions.