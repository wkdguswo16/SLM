import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import argparse
from models.retriever import Retriever, Config
from models.dataset import RetriverDataset, DataCollatorForSupervisedDataset
import os
from models.utils import load_config

def set_working_directory():
    """작업 디렉토리를 스크립트 파일이 위치한 디렉토리로 설정"""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    print(f"Working directory set to: {os.getcwd()}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(rank, args):
    current_gpu_index = rank
    torch.cuda.set_device(current_gpu_index)
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=current_gpu_index,
        init_method='env://'
    )
    if args.config_path is not None:
        print("Load config from {}".format(args.config_path))
        config = load_config(args.config_path)
    else:
        config = Config()
    model = Retriever(config)
    if args.checkpoint is not None:
        print("Load checkpoint from {}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    train_dataset = RetriverDataset(data_paths=args.data_paths)
    dist_train_samples = DistributedSampler(
        dataset=train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        seed=17
    )
    collate_batch = DataCollatorForSupervisedDataset(
        tokenizer=model.get_tokenizer())
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers, sampler=dist_train_samples, pin_memory=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=0.1)

    model = model.train().cuda(rank)
    model = DDP(model, device_ids=[rank])
    loss_sum = 0
    for epoch in range(args.epochs):
        if rank == 0:
            print("Epoch:{}/{} ==========>".format(epoch + 1, args.epochs))
        dist_train_samples.set_epoch(epoch)
        iters_sum = len(train_dataloader)
        for idx, inputs in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(rank)
            out = model(inputs)
            loss = out['key_loss']
            loss.backward()
            loss_sum = (loss_sum + loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if ((idx % args.log_interval == 0 and idx != 0) or idx == (iters_sum - 1)) and rank == 0:
                iters_last = args.log_interval if idx != (
                    iters_sum - 1) else iters_sum % args.log_interval
                print("iter {}/{}, loss: {}, lr:{}".format(idx, iters_sum,
                      loss_sum / (iters_last+1), optimizer.param_groups[0]['lr']))
                loss_sum = 0
        scheduler.step()

    if rank == 0:
        print("save model to {}".format(args.save_path))
        model = model.to("cpu")
        state_dict = model.module.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if "model." in key:
                state_dict.pop(key)
        torch.save(state_dict, args.save_path)


if __name__ == "__main__":
    set_working_directory()
    tasks = {"ag_news": [0, 4],
             "amazon": [4, 8],
             "dbpedia": [8, 12],
             "yahoo": [12, 16],
             "yelp": [16, 20]}
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('--epochs', default=3, type=int, metavar='N')
    parser.add_argument('--lr_step_size', default=1, type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument(
        '--save_path', default="", type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument(
        '--config_path', default="", type=str)
    parser.add_argument('--data_paths', default="ag", type=str)
    parser.add_argument('--master_port', default="8889", type=str)

    args = parser.parse_args()
    print(args)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.master_port
    args.world_size = args.gpus
    for task in tasks.keys():
        print(f"task trial: {task}")
        args.save_path = f"./dataset/config_6*4_{task}.pth"
        args.config_path =  f"./config_6*4_{task}.json"
        mp.spawn(
            train,
            nprocs=args.gpus,
            args=(args, )
        )
