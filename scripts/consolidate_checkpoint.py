import torch
import os
import sys
import json
from torch.distributed.fsdp import (
   FullyShardedDataParallel as FSDP,
   StateDictType,
   FullStateDictConfig,
)
from transformers import PretrainedConfig

def consolidate_fsdp_to_full(model_path, output_path=None, **model_kwargs):
   if output_path is None:
       output_path = f"{model_path}_consolidated"
   
   sys.path.append("/fsx/miquel/LongVU")
   from longvu.language_model.cambrian_qwen import CambrianQwenForCausalLM
   
   print(f"Loading config from {model_path}")
   config = PretrainedConfig.from_pretrained(model_path)
   
   print("Initializing model...")
   model = CambrianQwenForCausalLM(config)
   
   print("Loading FSDP checkpoint...")
   checkpoint = torch.load(os.path.join(model_path, "pytorch_model_fsdp.bin"), map_location='cpu')
   
   print("Loading state dict into model...")
   model.load_state_dict(checkpoint)
   
   print("Getting consolidated state dict...")
   state_dict = model.state_dict()
   
   print("\nVerifying shapes before saving:")
   for key, tensor in state_dict.items():
       if any(x in key for x in ['embed_tokens.weight', 'lm_head.weight', 'vision']):
           print(f"{key}: {tensor.shape}")
   
   # Create output directory
   os.makedirs(output_path, exist_ok=True)
   
   # Copy config files
   import shutil
   for file in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json']:
       src_file = os.path.join(model_path, file)
       if os.path.exists(src_file):
           shutil.copy2(src_file, os.path.join(output_path, file))
   
   print(f"\nSaving consolidated model to {output_path}")
   
   # Save in safetensors format
   from safetensors.torch import save_file
   
   # Split into chunks
   MAX_SIZE = 1024 * 1024 * 1024  # 1GB chunks
   chunks = {}
   current_chunk = {}
   current_size = 0
   
   # Sort keys to ensure consistent chunking
   sorted_keys = sorted(state_dict.keys())
   
   for k in sorted_keys:
       v = state_dict[k]
       if not isinstance(v, torch.Tensor):
           continue
       tensor_size = v.numel() * v.element_size()
       if current_size + tensor_size > MAX_SIZE:
           chunks[len(chunks)] = current_chunk
           current_chunk = {}
           current_size = 0
       current_chunk[k] = v
       current_size += tensor_size
   if current_chunk:
       chunks[len(chunks)] = current_chunk
   
   # Save chunks with metadata
   metadata = {"format": "pt"}
   for i, chunk in chunks.items():
       filename = f"model-{i+1:05d}-of-{len(chunks):05d}.safetensors"
       path = os.path.join(output_path, filename)
       print(f"Saving chunk {i+1}/{len(chunks)} to {filename}")
       save_file(chunk, path, metadata=metadata)
   
   # Create index file with metadata
   index = {
       "metadata": {"format": "pt"},
       "weight_map": {}
   }
   for i, chunk in chunks.items():
       filename = f"model-{i+1:05d}-of-{len(chunks):05d}.safetensors"
       for key in chunk.keys():
           index["weight_map"][key] = filename
   
   index_path = os.path.join(output_path, "model.safetensors.index.json")
   print(f"Saving index to {index_path}")
   with open(index_path, "w") as f:
       json.dump(index, f, indent=2)
   
   print("Verifying saved files...")
   saved_files = os.listdir(output_path)
   print(f"Files in {output_path}:")
   for f in saved_files:
       file_path = os.path.join(output_path, f)
       size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
       print(f"{f}: {size:.2f} MB")
   
   print("Done!")
   return output_path

if __name__ == "__main__":
   import torch.multiprocessing as mp
   mp.set_start_method('spawn', force=True)
   
   if len(sys.argv) > 1:
       model_path = sys.argv[1]
   else:
       model_path = "/path/to/your/checkpoint/"
   
   consolidated_path = consolidate_fsdp_to_full(model_path)