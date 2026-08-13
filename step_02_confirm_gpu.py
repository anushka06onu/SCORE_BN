"""## 2. Confirm GPU — T4 recommended

EDA and classical models do not need a GPU. Text-CNN/BiLSTM/BiGRU and transformer training should use the free T4.
"""

import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('CPU mode. Enable Runtime > Change runtime type > T4 GPU before deep models.')

