import timm
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
a = timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224
        )

b = timm.create_model(
            'vit_so400m_patch14_siglip_224', pretrained=True, num_classes=0, img_size=224
        )