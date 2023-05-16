from pathlib import Path

import matplotlib.pyplot as plt
import os
plt.style.use("ggplot")

import json

BASE_DIR = Path(__file__).parent.parent
LOG_PATH = os.path.join(BASE_DIR, 'results', 'TripletLoss_60_256_all', 'history.json')
with open(LOG_PATH, "r") as f:
    data = json.loads(f.read())
    epoch_train, loss = [x['epoch'] for x in data['train']], [x['loss'] for x in data['train']]
    epoch_val, precision = [x['epoch'] for x in data['val']], [x['accuracy'] for x in data['val']]

    print(f"Maximum Acc : {max(precision)}")
    print(f"Lowest Loss : {min(loss)}")

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(20, 5))

ax[0].plot(epoch_train, loss, 'r')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Train Loss')

ax[1].plot(epoch_val, precision, 'b')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Val Precision')

fig.tight_layout()
plt.show()
