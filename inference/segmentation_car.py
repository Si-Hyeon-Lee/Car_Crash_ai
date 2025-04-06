import torch
import cv2
import argparse
import numpy as np
from Models import Segmentation_model
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Perform image segmentation and save the overlaid results.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image.')
    parser.add_argument('--label_num', type=int,required=True,help='Segmentation Label Number. -1 means every label.')
    return parser.parse_args()

def main():
    args = parse_args()
    n_classes = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.label_num == -1:
        labels = ['label3', 'label2', 'label1', 'label0']
    else:
        labels = ['label'+str(args.label_num)]

    models = []
    for label in labels:
        model_path =  f'dist/assets/image/uploads/strict/save_models/{label}.pt'
    
        model = Segmentation_model(num_classes=n_classes).to(device)
        model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
    
        models.append(model)
    
    print('Loaded pretrained models!')
    
    img_path = args.input_path
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(img, (256, 256))
    
    img_input = original_img / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float().to(device)
    img_input = img_input.unsqueeze(0)
    
    overlaid_images = original_img.copy()
    
    for i, model in enumerate(models):
        output = model(img_input)
        img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
        img_output = img_output.squeeze(0) # (1, H, W)
    
        mask = np.zeros(original_img.shape, dtype=np.uint8)
        mask[img_output == 1] = np.array([255, 0, 0], dtype=np.uint8) # 빨간색
        
        overlaid_images[mask[:,:,0] > 0] = cv2.addWeighted(overlaid_images, 0.7, mask, 0.3, 0)[mask[:,:,0] > 0]
    
    overlaid_images = cv2.cvtColor(overlaid_images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, overlaid_images)
    print(f"Overlay image saved at {args.output_path}")

if __name__ == "__main__":
    main()
