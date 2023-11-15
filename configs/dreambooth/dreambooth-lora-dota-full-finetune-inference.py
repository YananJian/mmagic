_base_ = '../_base_/gen_default_runtime.py'

# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
#stable_diffusion_v15_url = '/data/huggingface_old/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/'

pretrained_model_path = '/data/yanan/output/yanan-dreambooth-dota-trainset-full-finetune/iter_160000_revised.pth'


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
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None),
    prior_loss_weight= 0.0,
    lora_config=lora_config)

