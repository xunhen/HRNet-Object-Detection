Add the iter_size
1 OptimizerHook(Hook) in optimizer of mmcv:
 ---------insert the iter_size;
2 build_dataloader() build_loader of mmdet:
----------insert the iter_size;
3 GroupSamplerIterSize in sampler of mmdet:
----------insert the iter_size;

about non-dist:
1  get_dist_info in runner.utils:
  --
      if not using_dist:
        initialized = None
  ---
using_dist