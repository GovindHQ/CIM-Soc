def build_model_with_cfg(model_cls, variant, pretrained, **kwargs):
    raise NotImplementedError(
        "The timm shim does not build models. Instantiate TinyViT directly "
        "and load the checkpoint with load_state_dict (see run_tinyvit.py)."
    )
