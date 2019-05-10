from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import GroupSampler, DistributedGroupSampler, GroupSamplerIterSize

# https://github.com/pytorch/pytorch/issues/973
# delete by wjc, because it's just for unix used
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     iter_size=None,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size,
                                          rank)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    elif iter_size is None or iter_size <= 1:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, imgs_per_gpu)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    else:
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSamplerIterSize(dataset, imgs_per_gpu, iter_size)
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
