"""Two-iteration engineering smoke test; never use for reported metrics."""

ModelParams = dict(
    extra_mark="sim",
    camera_extent=10,
    white_background=False,
)

OptimizationParams = dict(
    iterations=2,
    densify_from_iter=500,
    densify_until_iter=15000,
    opacity_reset_interval=3000,
)

ModelHiddenParams = dict(
    curve_num=20,
    ch_num=8,
    init_param=0.01,
)
