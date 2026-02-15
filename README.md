# KernelGPT

**KernelGPT** is a from-scratch implementation of a GPT (Generative Pre-trained Transformer) written in pure C, running directly on bare metal x86 hardware.

It boots a heavily modified vesion of [MooseOS](https://github.com/appleroll/moose-os) solely to train a neural network. No Linux, no Python, no PyTorch, no Standard Library (libc) — just raw pointers, math, and attention mechanisms.

It is heavily inspired off Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## Usage
Since this is a modified version of MooseOS, please check its [README](https://github.com/appleroll/moose-os/blob/main/README.md) for the usage guide.

## License
MIT License. Based on Andrej Karpathy's MicroGPT.
