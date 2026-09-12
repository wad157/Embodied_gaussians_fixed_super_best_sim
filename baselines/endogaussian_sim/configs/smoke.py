"""Two-step engineering smoke test; never use for reported baseline metrics."""

ModelParams = dict(
    extra_mark="sim",
    camera_extent=10,
    mode="binocular",
    white_background=False,
)

OptimizationParams = dict(
    coarse_iterations=2,
    iterations=2,
    densify_from_iter=500,
    densify_until_iter=0,
    pruning_from_iter=500,
    opacity_reset_interval=4000,
    position_lr_max_steps=4,
    lambda_dssim=0,
    lambda_lpips=0,
)

ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 8,
        "resolution": [16, 16, 16, 25],
    },
    multires=[1],
    defor_depth=0,
    net_width=16,
    plane_tv_weight=0,
    time_smoothness_weight=0,
    l1_time_planes=0,
    weight_decay_iteration=0,
    no_dx=True,
)
