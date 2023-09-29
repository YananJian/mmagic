# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
pretrained_model_path = '/data/dreambooth-lora/iter_10000_revised.pth'

val_prompts = [
    'a karameru cat in basket', 'a karameru cat on the mountain',
    'a karameru cat beside a swimming pool', 'a karameru cat on the desk',
    'a sleeping karameru cat', 'a screaming karameru cat'
]

lora_config = dict(target_modules=['to_q', 'to_k', 'to_v'])
model = dict(
    type='DreamBooth',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_config='configs/dreambooth/unet.json'),
        #from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    prior_loss_weight=0,
    lora_config=lora_config)
