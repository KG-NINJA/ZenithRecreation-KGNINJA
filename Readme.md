# Zenith Prompt Style Reproduction (Local CPU)

This repository demonstrates a successful local reproduction of the distinctive prompt/response style recently observed in *Zenith*, using only open-access tools and the [Phi-2 model](https://huggingface.co/microsoft/phi-2).

## ✨ What's Special?

This project replicates the rare dual-format output style:

=== BEST ===
Q: What is the capital of France?
A: Paris

=== WORST ===
Q: What is the capital of France?
A: Paris

perl
コピーする
編集する

This `BEST`/`WORST` side-by-side evaluation pattern was originally presented by a top-tier AI system. To our knowledge, this is the **first** open-source replication of that style, using only a CPU-based local setup.

## 🔧 How to Run

You need Python 3.10+ and the following packages:

```bash
pip install transformers
Then run the script:

bash
コピーする
編集する
python zenithZcpu.py
This will load Phi-2 and output BEST/WORST answer comparisons in Zenith-style.

📁 Files
zenithZcpu.py — Main script to generate Zenith-style output

sample_output.txt — Sample result from local run

requirements.txt — (Optional) for environment setup

💡 Why This Matters
This isn't just a curiosity — it's a symbolic moment:

An amateur, without any GPU, has re-created the rhythm of one of the world’s most advanced LLMs.

If Zenith represents a mountain peak of AI reasoning, this repo shows we can echo its silhouette with just a laptop and passion.

☕ Support My Exploration
If you found this interesting or inspiring, you can support me here:
👉 https://buymeacoffee.com/KGNINJA

#Zenith #LLM #PromptEngineering #KGNINJA #FirstContact
