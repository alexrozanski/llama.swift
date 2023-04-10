import sys
import time
import numpy
import torch

from sentencepiece import SentencePieceProcessor

# Simple dummy script to aid in debugging
print("Starting")
print(f'input: {sys.argv[1]}')
print(f'writing file...')
time.sleep(5)
fname_out = f"{sys.argv[1]}/ggml-model-1.bin"
with open(fname_out, "wb") as fout:
  fout.write(b" ")
