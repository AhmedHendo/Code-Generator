# Code Generator

A lightweight fine-tuning project focused on building a Python coding assistant from a pretrained Qwen language model. The goal is to teach the model to respond to coding prompts, generate Python functions, and assist with programming tasks using parameter-efficient fine-tuning.

## Project Overview

This repository explores a practical and cost-efficient approach to adapting a large language model for code generation without fully fine-tuning the entire base model.

The main experiment uses:

- Qwen 1.5B as the base model
- 4-bit quantization for memory-efficient loading
- LoRA adapters for efficient training
- Supervised fine-tuning on a Python instruction dataset
- Chat-style prompt formatting for instruction-following behavior

The result is a model that is better aligned for generating Python code from natural-language instructions.

## Why this project matters

Training a full LLM from scratch is expensive. Instead, this project demonstrates a more efficient workflow:

- keep the base model mostly frozen
- inject small trainable LoRA weights
- train only the adapter layers on a task-specific dataset
- preserve performance while drastically reducing compute and memory requirements

This makes the approach useful for experimentation, learning, and smaller deployment scenarios.

## Repository Structure

```text
Code Generator/
├── Part1.ipynb                 # Initial experimentation / baseline fine-tuning workflow
├── Part2/
│   ├── part_2_code.ipynb       # Main LoRA + Qwen fine-tuning notebook
│   └── qwen-sft-lora-final/    # Saved fine-tuned adapter/model artifacts
├── README.md                   # Project overview and usage notes
└── .gitignore                  # Optional repo hygiene file (if added later)
```

## What is being trained

The project uses a Python code dataset called `flytech/python-codes-25k`.

Each example is converted into a chat-style instruction format:

- System message: defines the role as a helpful coding assistant
- User message: contains the programming task or problem
- Assistant message: contains the expected Python solution

This helps the model learn to produce code in the style of a coding assistant.

## Training approach

The main workflow in `Part2/part_2_code.ipynb` includes:

1. Loading a Qwen base model
2. Applying 4-bit quantization with BitsAndBytes
3. Preparing the model for k-bit training
4. Adding LoRA adapters to transformer projection layers
5. Formatting the dataset into instruction-response pairs
6. Masking the prompt so the model learns only the assistant response
7. Training with SFTTrainer and LoRA configuration
8. Saving the final adapter/model weights for inference

## Model behavior

After fine-tuning, the model is expected to:

- explain Python concepts
- generate Python functions
- solve basic algorithmic problems
- respond with code based on natural-language prompts

This project is best suited for experimentation, learning, and small-scale code generation tasks.

## Main technologies used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- TRL
- BitsAndBytes
- Datasets
- W&B for tracking training (optional)

## Requirements

Install the following Python packages before running the notebooks:

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate wandb tqdm
```

You will also want a CUDA-enabled GPU if possible, because the training process is much faster and more practical on GPU hardware.

## How to run

1. Open the notebook in Jupyter or VS Code
2. Start with `Part1.ipynb` for the initial experimentation flow
3. Open `Part2/part_2_code.ipynb` for the main Qwen + LoRA training pipeline
4. Run the cells in order
5. After training, use the saved adapter/model files for inference or generation

## Example inference idea

After training, the model can be loaded and used like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen1.5B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
trained_model = PeftModel.from_pretrained(base_model, "./Part2/qwen-sft-lora-final")

prompt = "Write a Python function to reverse a string."
formatted_prompt = f"<|system|>\nYou are a helpful coding assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(trained_model.device)

with torch.no_grad():
    output = trained_model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Project status

This is an educational and research-oriented project focused on efficient fine-tuning for code generation. It demonstrates how a pretrained language model can be adapted for coding tasks with relatively modest resources.

## Future improvements

Possible next steps include:

- expanding the dataset with more real-world coding tasks
- evaluating on benchmark prompts and code correctness
- comparing LoRA rank and alpha settings
- testing larger Qwen models
- adding a small inference demo or web interface

## License

This project does not include a specific license file by default. If you plan to publish the repository publicly, consider adding an appropriate open-source license such as MIT or Apache 2.0.

## Acknowledgments

This project is built on top of open-source tools and models from the Hugging Face ecosystem, including Transformers, PEFT, TRL, and the Qwen model family.
