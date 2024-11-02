import errno
import os

import torch
import torch.distributed as dist

def all_gather(input_value):
    world_size = get_world_size()
    if world_size < 2:  # For one gpu
        return input_value
    with torch.no_grad():  # For multiple gpus
        output_value = [torch.zeros(size=input_value.shape, dtype=input_value.dtype) for _ in range(world_size)]
        dist.all_gather(output_value, input_value)
    return torch.concat(output_value, dim=0)
        
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # For one gpu
        return input_dict
    with torch.no_grad():  # For multiple gpus
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

def reduce_value(input_value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # For one gpu
        return input_value
    with torch.no_grad():  # For multiple gpus
        if isinstance(input_value, list):
            output_value = torch.stack(input_value, dim=0)
        else:
            output_value = input_value
        dist.all_reduce(output_value)
        if average:
            output_value = output_value / world_size
        
        if isinstance(input_value, list):
            return [v for v in output_value]
        else:
            return output_value

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # Only print in main process (rank = 0)
    setup_for_distributed(args.rank == 0)
