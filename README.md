# Medical-LLM-FineTuning 🏥 🤖

**Fine-Tuning Large Language Models for Structured Clinical Information Extraction.**

## 📌 Project Overview
This repository contains a research framework for fine-tuning LLMs (Qwen 2.5, Mistral, Llama 3) to perform **Information Extraction (IE)** on complex medical datasets.

The goal is to transform unstructured clinical notes into strict, validated **JSON outputs** using **Chain-of-Thought (CoT)** reasoning. The project focuses on model benchmarking, hyperparameter optimization, and efficient training strategies on limited hardware.

## 🚀 Key Features
* **Chain-of-Thought (CoT) Extraction:** Implements a "Reasoning + Extraction" approach to improve accuracy on complex medical cases.
* **Efficient Fine-Tuning:** Utilizes **QLoRA** and **Unsloth** for 2x faster training and 60% less VRAM usage.
* **Scientific Benchmarking:** A "Fair Comparison" framework to evaluate different architectures (Qwen vs. Mistral vs. Llama) under identical constraints.
* **Experiment Tracking:** Integrated with **Weights & Biases (WandB)** for real-time monitoring of training and evaluation loss.

## 🛠️ Tech Stack
* **Model Architecture:** Qwen 2.5 (1.5B/7B/14B), Mistral v0.3
* **Training Library:** Hugging Face `trl` (SFTTrainer), `peft` (LoRA), `unsloth`
* **Precision:** Bfloat16 (BF16) on NVIDIA RTX 6000 Ada
* **Tracking:** Weights & Biases (WandB)

## 📊 Methodology
1.  **Data Preparation:** Tokenization with dynamic chat templates (`<|im_start|>`, `[INST]`).
2.  **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) with Rank=64 and Alpha=32.
3.  **Evaluation:** Validation loss monitoring and structured JSON generation tests.