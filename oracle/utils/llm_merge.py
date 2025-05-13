from peft import PeftModel, PeftConfig
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, AutoConfig
import torch
import os
import shutil
from typing import Optional
from huggingface_hub import HfApi, login
import os
import torch
from openai import OpenAI
from tqdm import tqdm


IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def upload_to_hf(model_path: str, repo_id: str, commit_message: str = "Upload model"):
    login(token=os.environ.get("HF_TOKEN_W"))
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    print(f"Uploading model to {repo_id}...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message
    )


def delete_hf_model(repo_id: str, token: str):
    login(token=token)
    api = HfApi()
    api.delete_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model"
    )
    print(f"Successfully deleted repository: {repo_id}")
        

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def merge(base_model_id, adapter_path, merged_path):
    # Load PEFT config first to properly configure the base model
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model with correct 4-bit configuration
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True
    )
    
    model.eval()
    
    # Handle special tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )
    
    # Load and merge the PEFT adapter
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto",
        # No torch_dtype here as it's controlled by bnb_config
    )
    
    print("Merging weights...")
    # For 4-bit models, we need a specific merge approach
    merged_model = model.merge_and_unload(progressbar=True)
    
    # Ensure we're converting back to a format that can be saved
    print("Converting from 4-bit to higher precision for saving...")
    
    print(f"Saving merged model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print("Merge completed successfully!")


def quantize(merged_path, quant_path):
    model = AutoAWQForCausalLM.from_pretrained(merged_path, **{"low_cpu_mem_usage": True})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    base_model_id = "meta-llama/Llama-3.1-70B-Instruct"
    adapter_path = "JennyGan/70b-mp20-13500"
    merged_path = os.environ.get("HF_HOME") + "70b-mp20-13500-merged"
    
    # Ensure the output directory exists
    os.makedirs(merged_path, exist_ok=True)
    
    print("Merging weights...")
    merge(base_model_id, adapter_path, merged_path)
    
    # Uncomment if you want to also quantize the model
    # quant_path = merged_path + "-quant"
    # print(f"Quantizing model to: {quant_path}")
    # quantize(merged_path, quant_path)
    
    # Uncomment if you want to upload to HuggingFace
    # hf_repo_id = "JennyGan/crystalllm-3.1-70b"
    # print("Uploading to Hugging Face...")
    # upload_to_hf(
    #     model_path=merged_path,
    #     repo_id=hf_repo_id,
    #     commit_message="Upload merged model weights"
    # )