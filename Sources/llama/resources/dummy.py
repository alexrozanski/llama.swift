import sys
import time
import numpy
import torch

from sentencepiece import SentencePieceProcessor

print("Starting")
time.sleep(1)
print(f'input: {sys.argv[1]}')
print(f'writing file...')
fname_out = f"{sys.argv[1]}/ggml-model-1.bin"
with open(fname_out, "wb") as fout:
  fout.write(b" ")
