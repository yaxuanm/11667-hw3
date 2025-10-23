import os
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from calculator.utils import (
    load_asdiv,
    can_use_calculator,
    use_calculator,
    extract_label,
)
from tqdm.auto import tqdm
from datasets import Dataset

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("Warning: No GPU available, using CPU")
        return torch.device("cpu")

def main():
    """Initialize the pre-trained Pythia model"""
    device = get_device()
    print(f"Using device: {device}")
    
    # 先测试所有组件是否正常工作
    print("=" * 50)
    print("TESTING ALL COMPONENTS BEFORE TRAINING...")
    print("=" * 50)
    
    try:
        # 测试模型加载
        print("1. Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-1b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully")
        
        # 测试LoRA配置
        print("2. Testing LoRA configuration...")
        peft_config = LoraConfig(
            r=16,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        print("✓ LoRA configuration successful")
        
        # 测试数据集加载
        print("3. Testing dataset loading...")
        dataset = load_asdiv()
        print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
        # 测试tokenizer
        print("4. Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.padding_side = "right"
        tokenizer.pad_token = "<|padding|>"
        print("✓ Tokenizer loaded successfully")
        
        # 测试inference函数
        print("5. Testing inference function...")
        test_prefix = "Question: What is 2+2? Answer:"
        test_output = inference(model, tokenizer, test_prefix, calculator=False, max_tokens=5)
        print(f"✓ Inference test successful: {test_output}")
        
        print("=" * 50)
        print("ALL TESTS PASSED! Starting training...")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ ERROR in component testing: {e}")
        print("Fix the error before proceeding!")
        return

    # 检查模型是否已经存在
    if not os.path.exists("pythia-1b-asdiv"):
        print("=" * 50)
        print("STARTING TRAINING - This will take several minutes...")
        print("=" * 50)
        train(model, tokenizer, dataset["train"], batch_size=16, epochs=3)
        print("=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
    else:
        print("Model already exists, loading...")
        model = PeftModelForCausalLM.from_pretrained("pythia-1b-asdiv")
    
    print("Starting evaluation...")
    evaluate(model, tokenizer, dataset["test"])

    print("Done!")


def evaluate(
    model: PeftModelForCausalLM, tokenizer: AutoTokenizer, test_dataset: Dataset
):
    test_data = test_dataset.to_pandas()
    test_data["label"] = pd.to_numeric(test_data["label"])

    generations_calc = []
    labels_calc = []
    generations_no_calc = []
    labels_no_calc = []

    for prefix in tqdm(test_data["text"]):
        answer_calc = inference(model, tokenizer, prefix, calculator=True)
        answer_no_calc = inference(model, tokenizer, prefix, calculator=False)
        generations_calc.append(answer_calc)
        generations_no_calc.append(answer_no_calc)

        labels_calc.append(extract_label(answer_calc))
        labels_no_calc.append(extract_label(answer_no_calc))

    test_data["answer-calc"] = generations_calc
    test_data["answer-no-calc"] = generations_no_calc
    test_data["label-calc"] = labels_calc
    test_data["label-no-calc"] = labels_no_calc
    test_data.to_json("pythia-1b-asdiv/eval.jsonl", lines=True, orient="records")

    acc_calc = np.isclose(test_data["label-calc"], test_data["label"]).mean()
    acc_no_calc = np.isclose(test_data["label-no-calc"], test_data["label"]).mean()
    print(
        f"test accuracy with calculator: {acc_calc:.1%}",
    )
    print(
        f"test accuracy without calculator: {acc_no_calc:.1%}",
    )
    print("Done!")


def train(
    model: PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    grad_acc_steps: int = 1,
    batch_size: int = 32,
    epochs: int = 5,
) -> None:
    device = get_device()
    print(f"Training on device: {device}")
    
    # 先测试一个batch，确保没有错误
    print("Testing first batch...")
    try:
        tokenized_dataset = train_dataset.map(
            lambda x: {
                "input_ids": tokenizer.encode(x["text"] + x["target"])
                + [tokenizer.eos_token_id]
            }
        ).remove_columns(["text", "target", "label"])

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            shuffle=True,
        )
        
        # 测试第一个batch
        first_batch = next(iter(dataloader))
        first_batch = {k: v.to(device) for k, v in first_batch.items()}
        test_output = model(**first_batch)
        print("✓ First batch test successful!")
        
    except Exception as e:
        print(f"✗ Error in first batch test: {e}")
        raise e
    
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    step = 0
    for epoch_num in range(epochs):
        print(f"Starting epoch {epoch_num+1}/{epochs}")
        for batch in (pbar := tqdm(dataloader, desc=f"epoch {epoch_num+1}/{epochs}")):
            try:
                batch = {k: v.to(device) for k, v in batch.items()} 
                outputs = model(**batch)
                outputs["loss"].backward()

                if (step + 1) % grad_acc_steps == 0:
                    opt.step()
                    opt.zero_grad()

                pbar.set_postfix({"loss": outputs["loss"].item()})
                step += 1
            except Exception as e:
                print(f"✗ Error at step {step}, epoch {epoch_num+1}: {e}")
                raise e

    model.save_pretrained("pythia-1b-asdiv")


@torch.inference_mode(True)
def inference(
    model: PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    calculator: bool = True,
    max_tokens: int = 40,
) -> str:
    device = get_device()
    for i in range(max_tokens):
        if calculator and can_use_calculator(prefix):
            prefix = use_calculator(prefix)

        input_ids = tokenizer(prefix)["input_ids"]
        outputs = model(
            input_ids=torch.tensor([input_ids], dtype=torch.int64, device=device)
        )
        next_token_id = outputs["logits"][0][-1].argmax().item()
        if next_token_id == tokenizer.eos_token_id:
            break
        prefix = tokenizer.decode(input_ids + [next_token_id])
    return prefix


if __name__ == "__main__":
    main()
