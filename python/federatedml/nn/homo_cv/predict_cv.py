from federatedml.nn.homo_cv._torch import FedLightModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os 
from torch import nn
import torch
from PIL import Image
import numpy as np
import base64
# import cv2
from io import BytesIO



class My_Dataset(Dataset):
    def __init__(self, image_path, target):
        super().__init__()
        self.image_path = image_path
        self.target = target
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        path = self.image_path[index]
        img = Image.open(path).convert('RGB')
        resize_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        toTensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        label = self.target[index]
        if img.size[0] > 224:
            img = resize_transform(img)
        img = toTensor_transform(img)
        return img, label


# test
def test_model(test_loader, model, criterion, device):
    model.eval()
    
    true_labels = []
    pred_labels = []
    scores = []

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    print(f'size:{size}; num_batches:{num_batches}')
    losses, correct = 0, 0
    # log_loss = 0
    ################################# validation #################################
    with torch.no_grad():
        for batch, (x, y) in enumerate(tqdm(test_loader)):
            device = torch.device(device)
            # print(x)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # pred = nn.Softmax()(pred)
#             pred = pred[y.long().squeeze()]
            # print(f'pred:{pred}; y:{y.long().squeeze()}')
            # log_loss += logloss(pred, y.long().squeeze())
            loss = criterion(pred, y.long().squeeze()) 
            current = batch * len(x)
            scores += pred.tolist()
            y_pred, y_true = torch.argmax(pred, axis=1), y.long().squeeze()
            correct += (y_pred == y_true).type(torch.float).sum().item()
            loss, current = np.round(loss.item(), 5), batch * len(x)
            true_labels += y_true.detach().cpu().tolist()
            pred_labels += y_pred.detach().cpu().tolist()
            losses += loss
    correct /= size
    losses /= num_batches
    print(f'losses:{losses}\n')
    metrics = f"Test: Accuracy: {(100*correct):>0.2f}%, Avg loss: {losses:>5f} \n"
    return np.array(true_labels), np.array(pred_labels), np.array(scores), metrics





'''
job_id: job id
model_path: model path
res_path: save csv path
metric_path: mertic path
typ : predict or test
test_path: test data path
'''
def predict_cv(model_path, res_path, metric_path,  typ, test_path):
    model = FedLightModule.load_from_checkpoint(model_path)
    test_images_over = []
    test_labels_over = []
    if typ == 'test':
        with open(os.path.join(test_path, 'targets'), 'r') as f:
            line = f.readline()
            while line:
                filename = line.split(',')[0] + '.jpg'
                target = int(line.split(',')[-1].replace('\n',''))
                test_images_over.append(os.path.join(test_path, 'images', filename))
                test_labels_over.append(target)
                line = f.readline()
    elif typ == 'predict':
        with open(os.path.join(test_path, 'filenames'), 'r') as f:
            line = f.readline()
            while line:
                filename = line.replace('\n', '') + '.jpg'
                target = 0
                test_images_over.append(os.path.join(test_path, 'images', filename))
                test_labels_over.append(target)
                line = f.readline()
    test_dataset = My_Dataset(test_images_over, test_labels_over)
    test_loader = DataLoader(test_dataset, batch_size=32)
    loss_fn = nn.CrossEntropyLoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    true_labels, pred_labels, scores, metrics = test_model(test_loader, model, loss_fn, device)

    l = len(true_labels)
    if typ == 'test':
        with open(metric_path, 'w') as f:
            f.write(f'{metrics}\n')
        with open(res_path, 'w') as f:
            f.write('true_label,pred_label,score\n')
            for i in range(l):
                true_label, pred_label, score = true_labels[i], pred_labels[i], scores[i]
                f.write(f'{true_label},{pred_label},{score}\n')
    elif typ == 'predict':
        with open(res_path, 'w') as f:
            f.write('pred_label,score\n')
            for i in range(l):
                true_label, pred_label, score = true_labels[i], pred_labels[i], scores[i]
                f.write(f'{pred_label},{score}\n')



 
def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image).convert('RGB')
    return image



'''
model_path: 模型path
img_base64: 图片的路径
img_type: 0:宫颈癌，1结肠癌
device: device
'''
def predict_cv_one_sample(model_path, img_base64, img_type, device = 'cpu'):
    to_res = {0:{0:'Type1', 1:'Type2', 2: 'Type3'}, 1:{0:'阳性', 1:'阴性'}}
    model = FedLightModule.load_from_checkpoint(model_path)
    img = base64_pil(img_base64)
    resize_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    toTensor_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if img.size[0] > 224:
        img = resize_transform(img)
    img = toTensor_transform(img)

    # 必须要有这个，否则报错，简单来说就是给image升了一维
    img.unsqueeze_(0)
    x = img.to(device)
    # print(x.shape)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        # print(pred)
        y_pred = torch.argmax(pred, axis=1)
    # y_pred
    y_pred = y_pred.item()
    # print(type(y_pred))
    return to_res[img_type][y_pred]


def file_to_base64(path_file):    
    with open(path_file,'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf8')
        return image_base64

if __name__ == '__main__':
#     # # test predict_cv
    res_path = "/data/projects/fate/examples/results/123.csv"
    metric_path = "/data/projects/fate/examples/results/123.txt"
    typ = "test"
    path = "/data/projects/fate/examples/gwork/colon/test"
    module = "HomoCV"
    model_path = '/data/projects/fate/examples/gwork/colon/models/model.ckpt'
    predict_cv(model_path, res_path, metric_path,  typ, path)
    

    # # test predict_cv_one_sample
    # model_path = '/data/projects/fate/examples/gwork/colon/models/model_20221017.ckpt'
    # img_path = '/data/projects/fate/examples/gwork/colon/train_p1/images/colonca13.jpg'
    # img_type = 1
    # img_base64 = file_to_base64(img_path)
    # # print(img_base64)
    # with open('./base64.txt', 'w') as f:
    #     f.write(img_base64)
    # # # img = base64_pil(img_base64)
    # # # img.save('./111.jpg')
    # res = predict_cv_one_sample(model_path, img_base64, img_type, device = 'cpu')
    # print(res)

