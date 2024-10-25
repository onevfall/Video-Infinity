import torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import json
import os

from src.video_crafter import VideoCrafterPipeline, UNetVideoCrafter
from diffusers.schedulers import DPMSolverMultistepScheduler

from src.tools import DistController
from src.video_infinity.wrapper import DistWrapper

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Video Infinity Inference")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu_ids", type=str, help="Comma separated GPU IDs")
    args = parser.parse_args()
    return args

def init_pipeline(config):
    pipe = VideoCrafterPipeline.from_pretrained(
        'adamdad/videocrafterv2_diffusers',
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload(
        gpu_id=config["devices"][dist.get_rank() % len(config["devices"])],
    )
    pipe.enable_vae_slicing()
    return pipe

def run_inference(rank, world_size, config):
    dist_controller = DistController(rank, world_size, config)
    pipe = init_pipeline(config)
    dist_pipe = DistWrapper(pipe, dist_controller, config)
    start = time.time()

    pipe_configs=config['pipe_configs']
    plugin_configs=config['plugin_configs']

    prompt_id = int(rank / world_size * len(pipe_configs["prompts"]))
    prompt = pipe_configs["prompts"][prompt_id]

    start = time.time()
    dist_pipe.inference(
        prompt,
        config,
        pipe_configs,
        plugin_configs,
        additional_info={
            "full_config": config,
        }
    )
    print(f"Rank {rank} finished. Time: {time.time() - start}")

def main(config):
    size = len(config["devices"])
    processes = []

    if not os.path.exists(config["base_path"]):
        os.makedirs(config["base_path"])

    for rank, _ in enumerate(config["devices"]):
        p = mp.Process(target=run_inference, args=(rank, size, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def process_config(config, args):  # 添加 args 参数
    # Get prompts from environment variable
    prompt_group = os.getenv('PROMPT_GROUP')
    if prompt_group:
        # Split prompts by semicolon and clean up whitespace
        prompts = [p.strip() for p in prompt_group.split(';')]
        config['pipe_configs']['prompts'] = prompts
    
    # Get GPU IDs from command line arguments
    if args.gpu_ids:
        config['devices'] = [int(x) for x in args.gpu_ids.split(',')]
    
    return config

if __name__ == "__main__":
    mp.set_start_method("spawn")

    # 先解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 处理配置时传入 args
    config = process_config(config, args)
    
    main(config)
