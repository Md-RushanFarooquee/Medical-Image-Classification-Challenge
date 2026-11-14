# Image Classification Challenge

Build a model that assigns each **28×28 grayscale image** to one of **11 classes** (labelled `0–10`).

**Time limit:** 3 hours.

## Data

| Split | Images | Labels | Provided |
|---|---:|---|---|
| train | 13,375 | yes | yes |
| valid | 2,354 | yes | yes |
| test | hidden | hidden | no |

**The test set will not be provided.** Your submitted code is run on it by the organizers after the deadline. Use the valid split to estimate your score, but expect your test score to be somewhat lower: the test images are harder than the valid images, so models that generalise well will rank higher.

### Images per class

| Class | Train | Valid | Total |
|---:|---:|---:|---:|
| 0 | 1,112 | 183 | 1,295 |
| 1 | 767 | 264 | 1,031 |
| 2 | 1,920 | 274 | 2,194 |
| 3 | 1,493 | 202 | 1,695 |
| 4 | 1,082 | 131 | 1,213 |
| 5 | 595 | 90 | 685 |
| 6 | 1,077 | 156 | 1,233 |
| 7 | 604 | 100 | 704 |
| 8 | 710 | 248 | 958 |
| 9 | 688 | 237 | 925 |
| 10 | 3,327 | 469 | 3,796 |
| **Total** | **13,375** | **2,354** | **15,729** |

## File format (MNIST / IDX)

```text
data/train-images-idx3-ubyte.gz    uint8, shape (N, 28, 28), pixel values 0–255
data/train-labels-idx1-ubyte.gz    uint8, shape (N,), values 0–10
data/valid-images-idx3-ubyte.gz
data/valid-labels-idx1-ubyte.gz
```

Load them with the included `load_idx()` helper in `predict_template.py`, or any MNIST-format reader.

## Evaluation

- **Primary metric:** macro-averaged F1 over the 11 classes (every class counts equally, regardless of size).
- **Tie-breaker:** accuracy (more is better), then model size (smaller is better).
- Final score will be decided by the organizers.

## Submission

Submit **one folder** (or a `.zip` of it) named after your team, e.g. `team_falcon/`, containing:

1. `predict.py` (required; must be a `.py` file— notebooks are not run) defining:

   ```python
   def predict(images: np.ndarray) -> np.ndarray:
       # images: uint8 array, shape (N, 28, 28), values 0–255
       # return: int array, shape (N,), values 0–10
   ```

   It must load its weights from files in the same folder (use `os.path.dirname(__file__)`).

2. Your weight file(s), e.g. `model.pt`.
3. Your training code: `train.py` or the notebook you trained with.
4. Submit only the final model, not the checkpoints.
5. The final zip will be submitted to `@gmail.com` with title **"ML Hackathon Submission Group No : <no.>"**.

## Rules

1. Train **only** on the data in this kit. No external datasets, and no pretrained weights other than standard ImageNet weights.
2. One submission per team; the last upload before the deadline counts.
3. Training code is reviewed. Submissions that break the rules, or whose score cannot be reproduced from the training code, are disqualified.
4. A submission that fails to run (error, wrong output, over the time limit) scores zero. Run `check_submission.py` before uploading.
5. Do not share code or predictions between teams.

## Getting started

1. First, change the runtime type to GPU: **Runtime → Change runtime type → GPU**.
2. The data is available in Google Drive. Mount Google Drive first:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. The data path will be `/content/drive/MyDrive/data`.
4. Save model checkpoints and the final model to `/content/drive/MyDrive/models`.
5. You have full access to Gemini Pro and a Google Pro account. Use responsibly.

### Load the data

The following code is for PyTorch. If using TensorFlow, change it accordingly.

```python
import gzip, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA = '/content/drive/MyDrive/data'


def load_idx(path):
    'Read an MNIST/IDX .gz file into a numpy array.'
    with gzip.open(path, "rb") as f:
        _, _, dtype, ndim = struct.unpack(">BBBB", f.read(4))
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_tensor(a):
    return torch.from_numpy(a.copy()).float().div(255).unsqueeze(1).to(dev)


xtr = to_tensor(load_idx(f'{DATA}/train-images-idx3-ubyte.gz'))
ytr = torch.from_numpy(
    load_idx(f'{DATA}/train-labels-idx1-ubyte.gz').astype(np.int64)
).to(dev)

xva = to_tensor(load_idx(f'{DATA}/valid-images-idx3-ubyte.gz'))
yva = load_idx(f'{DATA}/valid-labels-idx1-ubyte.gz')

print(dev, xtr.shape, xva.shape)
```

Make sure your `predict.py` works on `valid_set` before final submission.
