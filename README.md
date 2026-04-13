# QLoRA instruction fine-tuning

This repository contains a completed Jupyter notebook that fine-tunes a large language model for instruction following using **4-bit quantization** and **LoRA** (QLoRA), so training fits constrained GPUs (for example a free Google Colab T4 session).

## Contents

| File | Description |
|------|-------------|
| [`Take_Home_Coding_Task_Finished.ipynb`](Take_Home_Coding_Task_Finished.ipynb) | End-to-end walkthrough: dataset loading, model setup, custom training loop, batched text generation, validation loss vs. training set size, and chat-template handling when switching models. |

The notebook targets **Mistral-7B-Instruct-v0.2** with `bitsandbytes` and `peft`, and uses the [`yizhongw/self_instruct`](https://huggingface.co/datasets/yizhongw/self_instruct) instruction dataset from the Hugging Face Hub.

## Requirements

Install the same stack as the first notebook cell (Python 3.10+ recommended):

```bash
pip install accelerate peft transformers bitsandbytes datasets trl torch huggingface_hub
```

You also need a CUDA-capable GPU with enough VRAM for 4-bit 7B inference plus LoRA training (exact headroom depends on batch size and sequence length).

## How to run

1. **Google Colab**  
   Upload or open the notebook in Colab, enable a GPU runtime, and run cells top to bottom. The notebook includes `pip install` for dependencies.

2. **Local Jupyter**  
   Clone the repo, install the packages above, open the notebook in Jupyter or VS Code, and run all cells.

Some Hub models or datasets may require a [Hugging Face token](https://huggingface.co/settings/tokens). If access is denied, log in with `huggingface-cli login` locally or use Colab secrets / `notebook_login()` as in the notebook.

## What the notebook implements

- Load and preprocess an instruction-following dataset.
- Load a causal LM with 4-bit quantization and attach LoRA adapters.
- A small **custom trainer** with optimizer, scheduler, and next-token prediction loss.
- **Batched `generate()`** built on `model.forward()` (EOS / max length stopping).
- **Experiment**: vary training subset size and plot validation loss.
- **Chat templates**: use the tokenizer’s template API so prompts stay correct when the base model changes (with an example using another model family).

## License

Refer to the licenses of the bundled notebook, Hugging Face models (`mistralai/Mistral-7B-Instruct-v0.2`, etc.), and datasets you load from the Hub.
