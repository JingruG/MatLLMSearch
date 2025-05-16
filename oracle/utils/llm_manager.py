import torch
from vllm import LLM, SamplingParams
from openai import OpenAI
import os
# from vllm.model_executor.adapters import lora
# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

from peft import PeftModel, PeftConfig, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import shutil
from typing import Optional
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
import concurrent.futures
import threading
import time
import json


class LLMManager:
    def __init__(self, base_model, tensor_parallel_size=1, gpu_memory_utilization=0.8, temperature=1.0, max_tokens=200, seed=42):
        hf_token = os.environ.get("HF_TOKEN_W")
        login(token=hf_token)
        self.base_model = base_model
        self.seed = seed
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm, self.tokenizer = self._initialize_llm(tensor_parallel_size, gpu_memory_utilization)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens)
        self.lora_request = None


    def _load_api_keys(self):
        """Load API keys from a JSON file"""
        api_keys_file = "oracle/utils/api_keys.json"
        
        try:
            # Load keys from file
            with open(api_keys_file, 'r') as f:
                data = json.load(f)
                keys = data.get("deepseek_keys", [])
                
            if keys:
                print(f"Loaded {len(keys)} API keys from {api_keys_file}")
                return keys
        except Exception as e:
            print(f"Error loading API keys: {str(e)}")
        
        # Fallback to environment variable
        key = os.environ.get("DS_API_KEY")
        if key:
            return [key]
        
        raise ValueError("No API keys found. Please create an api_keys.json file or set DS_API_KEY environment variable")
        
    def _get_next_api_key(self):
        """Rotate to the next API key"""
        if not self.api_keys:
            raise ValueError("No API keys available")
            
        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_index]
        
        # Reinitialize the client with the new key
        if "deepseek" in self.base_model:
            self.llm = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=new_key
            )
            
        print(f"Switched to DS API key #{self.current_key_index + 1}")
        return new_key
        
    def smart_tokenizer_and_embedding_resize(self, 
        special_tokens_dict, 
        llama_tokenizer, 
        model,
    ):
        """Resize tokenizer and embedding.
    
        Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
        """
        num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(llama_tokenizer))
    
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
    
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
    def _initialize_llm(self, tensor_parallel_size, gpu_memory_utilization):
        if "gpt" in self.base_model:
            return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), None
        elif "deepseek" in self.base_model:
            self.api_keys = self._load_api_keys()
            self.current_key_index = 0
            return OpenAI(base_url="https://openrouter.ai/api/v1",
                          api_key=self.api_keys[self.current_key_index]), None
        elif "crystalllm" in self.base_model:
            # IGNORE_INDEX = -100
            # MAX_LENGTH = 2048
            # DEFAULT_PAD_TOKEN = "[PAD]"
            # DEFAULT_EOS_TOKEN = "</s>"
            # DEFAULT_BOS_TOKEN = "<s>"
            # DEFAULT_UNK_TOKEN = "<unk>"
            # model_string = "meta-llama/Llama-3.1-70B-Instruct"
            # model = LlamaForCausalLM.from_pretrained(
            #     model_string,
            #     load_in_4bit=True,
            #     device_map="auto",
            # )
            # tokenizer = AutoTokenizer.from_pretrained(
            #     model_string,
            #     model_max_length=MAX_LENGTH,
            #     trust_remote_code=True,
            #     padding_side="right",
            #     use_fast=False)
            # model.eval()
            # special_tokens_dict = dict()
            # if tokenizer.pad_token is None:
            #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
            # if tokenizer.eos_token is None:
            #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            # if tokenizer.bos_token is None:
            #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            # if tokenizer.unk_token is None:
            #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
            # self.smart_tokenizer_and_embedding_resize(
            #     special_tokens_dict=special_tokens_dict,
            #     llama_tokenizer=tokenizer,
            #     model=model,
            # )
            # model = PeftModel.from_pretrained(model, "JennyGan/70b-mp20-9000", device_map="auto")
            # return model, tokenizer

            vllm_model = LLM(
                model="meta-llama/Llama-3.1-70B-Instruct",
                tensor_parallel_size=tensor_parallel_size,  # Number of GPUs for tensor parallelism
                gpu_memory_utilization=gpu_memory_utilization,  # Optional, default is 0.9
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                trust_remote_code=True,
                dtype="half",  # Use half precision (fp16)
                adapter_path="JennyGan/70b-mp20-9000",  # Path to your adapter weights
                # You can add more vLLM-specific parameters here
                enforce_eager=False,  # Set to False to use CUDA graphs for faster execution
                max_num_seqs=32,  # Adjust based on your throughput needs
                max_num_batched_tokens=8192,  # Adjust based on your GPU memory
                quantization="awq",  # Options: awq, gptq, none - try different ones
            )
            
            # For tokenization consistency, we still need the HF tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_string)
            
            return vllm_model, tokenizer

            # return LLM(
            #     model=self.base_model,
            #     dtype=torch.float16,
            #     tensor_parallel_size=tensor_parallel_size,
            #     gpu_memory_utilization=gpu_memory_utilization,
            #     max_model_len=self._get_max_token_length(),
            #     seed=self.seed,
            #     # quantization="bitsandbytes",
            #     # load_format="bitsandbytes",
            #     # enforce_eager=True,
            #     trust_remote_code=True,
            #     max_num_seqs=8,
            # )
        elif "lora" in self.base_model:
            # model = AutoModelForCausalLM.from_pretrained(self.base_model, 
            #                                 device_map="auto",
            #                                 torch_dtype=torch.bfloat16)

            # llm = LLM(model="meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.90)
            # lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, "JennyGan/70b-mp20-14000") 

            if '70b' in self.base_model:
                base_model_id = "meta-llama/Llama-3.1-70B-Instruct"
                lora_path = snapshot_download(repo_id="JennyGan/70b-mp20-14000")
            else:
                base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
                lora_path = "/local2/anonymous/M2Hub/crystal-text-llm/exp/7b-test-run/checkpoint-7500"             
            llm = LLM(model=base_model_id, enable_lora=True, 
                dtype=torch.float16,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                trust_remote_code=True,
                max_num_seqs=8,)
            self.lora_request = LoRARequest("lora", 1, lora_path)
            tokenizer = AutoTokenizer.from_pretrained(lora_path)
            return llm, tokenizer
        else:
            model = LLM(
                model=self.base_model,
                dtype=torch.float16,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=self._get_max_token_length(),
                seed=self.seed,
                trust_remote_code=True,
                max_num_seqs=8,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            return model, tokenizer

    def _get_max_token_length(self):
        if '70b' in self.base_model:
            return 11000
        elif 'mistral' in self.base_model:
            return 32000
        return 11000
        
    def generate(self, prompts):
        prompts = [[
            { "role": "user", "content": prompt },
         ] for prompt in prompts]
        if self.tokenizer is not None:
            prompts = [self.tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]
        if "gpt" in self.base_model:
            results = []
            for i, prompt in enumerate(tqdm(prompts, desc="GPT Generation", unit="prompt")):
                result = self.generate_gpt(prompt)
                results.append(result)
            return results
        elif "deepseek" in self.base_model:
            results = []
            for i, prompt in enumerate(tqdm(prompts, desc="DeepSeek Generation", unit="prompt")):
                result = self.generate_deepseek(prompt)
                results.append(result)
            return results
        elif "flowmm" in self.base_model:
            results = self.llm.generate(prompts)
        elif "crystalllm" in self.base_model:
            batch_size = 10  # Adjust this value based on your GPU memory
            all_gen_strs = []
            for i in tqdm(range(0, len(prompts), batch_size), desc="Finetuned Model Generation", unit="batch"):
                batch_prompts = list(prompts)[i:i+batch_size]
                batch = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt",
                    truncation=True, 
                    padding=True 
                )
                batch = {k: v.cuda() for k, v in batch.items()}
                generate_ids = self.llm.generate(
                    **batch,
                    do_sample=True,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature, 
                    top_p=0.9, 
                )
                batch_gen_strs = []
                for ids in generate_ids:
                    decoded = self.tokenizer.decode(
                        ids, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    batch_gen_strs.append(decoded)
                all_gen_strs.extend(batch_gen_strs)
                del batch, generate_ids
                torch.cuda.empty_cache()
            return all_gen_strs
        else:
            results = self.llm.generate(prompts, self.sampling_params, lora_request=self.lora_request)
            return [output.text for result in results for output in result.outputs]
        
    def generate_gpt(self, message):
        response = self.llm.chat.completions.create(
            model=self.base_model,
            messages=message,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()


    def generate_deepseek(self, message, timeout_seconds=180, max_retries=2, max_key_rotations=3):
        """Generate response with detailed error handling and debugging"""
        
        # Use a shared result container
        result_container = {"result": "", "completed": False}
        key_rotations = 0
        
        def _generate():
            nonlocal key_rotations
            retries = 0
            while retries <= max_retries and not result_container["completed"]:
                try:
                    print(f"Sending request with API key #{self.current_key_index + 1}")
                    response = self.llm.chat.completions.create(
                        extra_body={},
                        model="deepseek/deepseek-chat-v3-0324:free", 
                        messages=message,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    
                    if not result_container["completed"]:
                        try:
                            result_container["result"] = response.choices[0].message.content.strip()
                            result_container["completed"] = True
                        except (AttributeError, TypeError) as e:
                            print(f"Error extracting response content: {str(e)}")
                            
                            # Try rotating API key
                            if key_rotations < max_key_rotations and len(self.api_keys) > 1:
                                key_rotations += 1
                                self._get_next_api_key()
                                print("Trying with new API key...")
                                time.sleep(1)
                            else:
                                retries += 1
                                if retries <= max_retries:
                                    print(f"Retrying ({retries}/{max_retries+1})...")
                                    time.sleep(1)
                                else:
                                    result_container["completed"] = True
                except Exception as e:
                    if not result_container["completed"]:
                        print(f"Error in generation: {str(e)[:100]}...")
                        
                        # Try rotating API key
                        if key_rotations < max_key_rotations and len(self.api_keys) > 1:
                            key_rotations += 1
                            self._get_next_api_key()
                            print("Trying with new API key...")
                            time.sleep(2)
                        else:
                            retries += 1
                            if retries <= max_retries:
                                print(f"Retrying ({retries}/{max_retries+1})...")
                                time.sleep(1)
                            else:
                                result_container["completed"] = True
        
        # Create and start thread
        thread = threading.Thread(target=_generate)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        start_time = time.time()
        while not result_container["completed"] and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)
        
        # Mark as completed even if it's a timeout
        result_container["completed"] = True
        
        # Return result or empty string
        if thread.is_alive():
            print(f"Request timed out after {timeout_seconds} seconds")
        
        return result_container["result"]