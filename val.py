import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from glob import glob
from func.utils import get_loader, load_model, evaluate
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
import warnings
import torch.nn.functional as F
import cv2
import shutil
warnings.filterwarnings("ignore", category=FutureWarning)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="style", type=str, help="swin/s0/s2")
    parser.add_argument("--data_path", default="/code/Font/to_do/word1", type=str, help="path to val data")
    parser.add_argument("--port", default=8673, type=int, help="port of dist")
    parser.add_argument("--batch_size", default=128, type=int, help="") # 4, 128, 32, 56, 56
    parser.add_argument("--num_workers", default=6, type=int, help="")
    args = parser.parse_args()

    return args

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

if __name__ == '__main__':
    
    # get args
    args = create_parser()
    args.n_classes = 17350
    args.dict_path = './cfgs/char_classes_17350.json'
    args.model_path = './cfgs/content/model_0.989.pth'

    if args.mode == 'content':
        args.n_classes = 4807
        args.model_path = './cfgs/content/model_0.982_.pth'
        args.dict_path = './cfgs/char_classes_4807.json'
    elif args.mode == 'style':
        args.n_classes = 173
        # args.model_path = './cfgs/style.pt'
        args.model_path = './cfgs/content.pt'
        args.dict_path = './cfgs/font_classes_173.json'

    # if args.mode == 's0':
    #     args.model_path = './cfgs/content/model_s0_0.984_.pth'
    # elif args.mode == 's2':
    #     args.model_path = './cfgs/content/model_s2_0.987.pth'
    # elif args.mode == 'swin':
    #     args.model_path = './cfgs/content/model_swin_0.989.pth'
    
    model = load_model(args)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = glob(os.path.join(args.data_path, '*.png'))

    transform = Compose([
        ToPILImage(),
        Resize((224, 224)), 
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for idx in range(15,len(images)):
        img_cv = cv2.imread(images[idx])
        print(images[idx])
        img_tensor = transform(img_cv).unsqueeze(0).to(device)
        

        feature_output = model.forward_features(img_tensor)
        mat_1 = gram_matrix(feature_output)
        
        for idx_t in range(idx+1, len(images)):
            img_cv_t = cv2.imread(images[idx_t])
            img_tensor_t = transform(img_cv_t).unsqueeze(0).to(device)
            

            feature_output_t = model.forward_features(img_tensor_t)
            mat_1_t = gram_matrix(feature_output_t)

            loss_1 = F.mse_loss(mat_1, mat_1_t)
            loss_2 = F.mse_loss(feature_output, feature_output_t)
            if (loss_2 < 1.5):
                class_num = 'class8'
                if not os.path.exists(os.path.join(args.data_path, class_num)):
                    os.mkdir(os.path.join(args.data_path, class_num))
                path = os.path.join(os.path.dirname(images[idx_t]), class_num)
                dst_path = os.path.join(path, os.path.basename(images[idx_t]))
                # print(dst_path)
                shutil.move(images[idx_t], dst_path)
                

                print(os.path.basename(images[idx_t]), loss_1, loss_2)


        break

        # print(feature_output)
        # print(feature_output.size())
        # print(mat_1.size())
    
    
    # val_loader = get_loader(args)
    # for i, (img, label) in enumerate(val_loader):
    #     print(img)
    #     print(img.size())
    #     img = img.to(device)
    #     feature_output = model.forward_features(img)
    #     print(feature_output.size())
    #     break

    # acc = evaluate(model, val_loader)
    # print('average accuracy: {}%'.format(acc*100))

