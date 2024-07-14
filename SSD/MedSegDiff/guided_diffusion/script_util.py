import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel_newpreview, UNetModel_v1preview, EncoderUNetModel

NUM_CLASSES = 2

{
# def classifier_defaults():
#     """
#     Defaults for classifier models.
#     """
#     return dict(
#         image_size=64,
#         classifier_use_fp16=False,
#         classifier_width=128,
#         classifier_depth=2,
#         classifier_attention_resolutions="32,16,8",  # 16
#         classifier_use_scale_shift_norm=True,  # False
#         classifier_resblock_updown=True,  # False
#         classifier_pool="spatial",
#     )

# def classifier_and_diffusion_defaults():
#     res = classifier_defaults()
#     res.update(diffusion_defaults())
#     return res
}

#::diffusion_defaults
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

#:model_defaults+diffusion_defaults:dictionary(o)
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        in_ch = 5,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        dpm_solver = False,
        version = 'new',
    )
    res.update(diffusion_defaults())
    return res

#arguments::model,gaussian
def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    in_ch,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    dpm_solver,
    version,
):
    model = create_model(
        image_size,  # small size as x0,x1,... :same size, huge model unlike VAE
        num_channels, # internal unet dimension: [512,264,128] like this
        num_res_blocks, # no. of residul block in one layer
        channel_mult=channel_mult, # multiple image channel ns
        learn_sigma=learn_sigma, # learn variance or not
        class_cond=class_cond, # class conditioning: adding extra info for a class
        use_checkpoint=use_checkpoint, # optimization(save memory/more time): do not save activation function on foreward pas recompute on backprop
        attention_resolutions=attention_resolutions, # at resolution 16x16 attention will be done after how many downsampling attention will be done
        in_ch = in_ch,
        num_heads=num_heads, # how many attentions
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,# ways of conditioning image and time step
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        version = version,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        dpm_solver=dpm_solver,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

#2 exploring
def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    in_ch=4,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    version = 'new',
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    
    # after how many downsampling is attention performed
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel_newpreview(
        image_size=image_size, # size of image
        in_channels=in_ch,     # number of channels
        model_channels=num_channels,
        out_channels=2,#(3 if not learn_sigma else 6), one for noise and one for varience 6 for RGB images
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        # num_classes=(NUM_CLASSES if class_cond else None),
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    ) if version == 'new' else UNetModel_v1preview(
        image_size=image_size,
        in_channels=in_ch,
        model_channels=num_channels,
        out_channels=2,#(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        # num_classes=(NUM_CLASSES if class_cond else None),
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
    

def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=2,#1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


# def sr_model_and_diffusion_defaults():
#     res = model_and_diffusion_defaults()
#     res["large_size"] = 256
#     res["small_size"] = 64
#     arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
#     for k in res.copy().keys():
#         if k not in arg_names:
#             del res[k]
#     return res


# def sr_create_model_and_diffusion(
#     large_size,
#     small_size,
#     class_cond,
#     learn_sigma,
#     num_channels,
#     num_res_blocks,
#     num_heads,
#     num_head_channels,
#     num_heads_upsample,
#     attention_resolutions,
#     dropout,
#     diffusion_steps,
#     noise_schedule,
#     timestep_respacing,
#     use_kl,
#     predict_xstart,
#     rescale_timesteps,
#     rescale_learned_sigmas,
#     use_checkpoint,
#     use_scale_shift_norm,
#     resblock_updown,
#     use_fp16,
# ):
#     model = sr_create_model(
#         large_size,
#         small_size,
#         num_channels,
#         num_res_blocks,
#         learn_sigma=learn_sigma,
#         class_cond=class_cond,
#         use_checkpoint=use_checkpoint,
#         attention_resolutions=attention_resolutions,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         dropout=dropout,
#         resblock_updown=resblock_updown,
#         use_fp16=use_fp16,
        
#     )
#     diffusion = create_gaussian_diffusion(
#         steps=diffusion_steps,
#         learn_sigma=learn_sigma,
#         noise_schedule=noise_schedule,
#         use_kl=use_kl,
#         predict_xstart=predict_xstart,
#         dpm_solver = dpm_solver,
#         rescale_timesteps=rescale_timesteps,
#         rescale_learned_sigmas=rescale_learned_sigmas,
#         timestep_respacing=timestep_respacing,
#     )
#     return model, diffusion


# def sr_create_model(
#     large_size,
#     small_size,
#     num_channels,
#     num_res_blocks,
#     learn_sigma,
#     class_cond,
#     use_checkpoint,
#     attention_resolutions,
#     num_heads,
#     num_head_channels,
#     num_heads_upsample,
#     use_scale_shift_norm,
#     dropout,
#     resblock_updown,
#     use_fp16,
# ):
#     _ = small_size  # hack to prevent unused variable

#     if large_size == 512:
#         channel_mult = (1, 1, 2, 2, 4, 4)
#     elif large_size == 256:
#         channel_mult = (1, 1, 2, 2, 4, 4)
#     elif large_size == 64:
#         channel_mult = (1, 2, 3, 4)
#     else:
#         raise ValueError(f"unsupported large size: {large_size}")

#     attention_ds = []
#     for res in attention_resolutions.split(","):
#         attention_ds.append(large_size // int(res))

#     return SuperResModel(
#         image_size=large_size,
#         in_channels=3,
#         model_channels=num_channels,
#         out_channels=(3 if not learn_sigma else 6),
#         num_res_blocks=num_res_blocks,
#         attention_resolutions=tuple(attention_ds),
#         dropout=dropout,
#         channel_mult=channel_mult,
#         num_classes=(NUM_CLASSES if class_cond else None),
#         use_checkpoint=use_checkpoint,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=resblock_updown,
#         use_fp16=use_fp16,
#     )

#2 exploring
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    dpm_solver = False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    # getting betas for particular noise shedule and steps
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # type of loss function used
    # normal loss of 1st paper
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    # hybrid MSE
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # reducing number of steps in sampling
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        # reduce no. of steps in sampling
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        # deciding to predict error or image
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # type of variance
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        dpm_solver=dpm_solver,
        rescale_timesteps=rescale_timesteps,
    )

# parser,dic:adding arguments to parser: 
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

# attributes,keys:{k:attr}:dic
def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

# bool/str:check bool:bool
# checking for all types of strings which are bool
#D
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
