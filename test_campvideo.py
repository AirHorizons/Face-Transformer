import torch
import torch.nn as nn
import sys
from vit_pytorch import ViTs_face, ViT_face
import sklearn
import cv2
import os

def main():
    MULTI_GPU = False
    DEVICE = torch.device("cuda:0")
    NUM_CLASS = 93431

    model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='CosFace',
            GPU_ID= DEVICE,
            num_class=NUM_CLASS,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    # model = ViTs_face(
    #         loss_type='CosFace',
    #         GPU_ID=DEVICE,
    #         num_class=NUM_CLASS,
    #         image_size=112,
    #         patch_size=8,
    #         ac_patch_size=12,
    #         pad=4,
    #         dim=512,
    #         depth=20,
    #         heads=8,
    #         mlp_dim=2048,
    #         dropout=0.1,
    #         emb_dropout=0.1
    #     )

    model_root = '/workspace/Face-Transformer/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'

    model.load_state_dict(torch.load(model_root))

    test_image = cv2.imread('/workspace/extracted_frames/bMpIikYOkrQ/frame_146.jpg')
    test_image = cv2.resize(test_image, (112, 112))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.tensor(test_image).unsqueeze(0).float()
    test_image = test_image.to(DEVICE)

    model = model.to(DEVICE)

    model.eval()

    with torch.no_grad():
        output = model(test_image)
        print(torch.argmax(output), output[0,373].item())
        print(output[0][:10])

if __name__ == '__main__':
    main()