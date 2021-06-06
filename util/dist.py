# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import time
import subprocess
import torch
from torch import distributed as dist
import functools
print = functools.partial(print, flush=True)

def init_process_group(args):
    if args.dist_url is None:
        master_ip = subprocess.check_output(['hostname', '--fqdn']).decode('utf-8')
        master_ip = str(master_ip).strip()
        args.dist_url = 'tcp://{}:23457'.format(master_ip)
        if args.rank == 0:
            print(args.dist_url)

        # ------------------------hack for multi-machine training -------------------- #
        if args.world_size > 8:
            ip_add_file = './' + args.output_dir + '/ip_add.txt'
            if args.rank == 0:
                with open(ip_add_file, 'w') as ip_add:
                    ip_add.write(args.dist_url)
            else:
                while not os.path.exists(ip_add_file):
                    time.sleep(0.5)

                with open(ip_add_file, 'r') as ip_add:
                    dist_url = ip_add.readline()
                args.dist_url = dist_url

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    print('Rank {} initialization finished.'.format(args.rank))
    synchronize()

    if args.rank == 0:
        if os.path.exists('./' + args.output_dir + '/ip_add.txt'):
            os.remove('./' + args.output_dir + '/ip_add.txt')


def reduce_tensor(tensor):
    rt = tensor.clone()
    try:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    except AssertionError:
        pass
    rt /= dist.get_world_size()
    return rt


def configure_nccl():
    """Configure multi-machine environment variables.

    It is required for multi-machine training.
    """
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_IB_DISABLE"] = "1"

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def dist_collect(x):
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def dist_collect_grad(x):
    gpu_id = dist.get_rank()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    out_list[gpu_id] = x
    return torch.cat(out_list, dim=0)
