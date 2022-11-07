from mimetypes import init
from federatedml.nn.backend.pytorch.MMBT.image import ImageEncoderDenseNet
from federatedml.nn.backend.pytorch.MMBT.mmbt_config import MMBTConfig
from federatedml.nn.backend.pytorch.MMBT.mmbt import MMBTForClassification
from federatedml.nn.backend.pytorch.MMBT.mmbt_utils import JsonlDataset, collate_fn, get_multiclass_criterion
from federatedml.nn.homo_mm._torch import FedLightModule
import argparse
import json
import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SequentialSampler

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from PIL import Image
import numpy as np
import base64
from io import BytesIO
import numpy as np

def get_image_transforms():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def load_examples(tokenizer, wandb_config, data_path, img_dir):
    """

    :param tokenizer: BERT tokenizer of choice
    :param wandb_config: wandb.config, which needs to contain file names of validation, test, and train files
    :param data_dir: Path to jsonl data directory e.g. "data/json"
    :param img_dir: Path to image directory e.g. "NLMCXR_png_frontal"
    :return: JasonlDataset derived from Torch Dataset class
    """

    path = data_path 
    
    img_transforms = get_image_transforms()

    labels = [0,1]

    dataset = JsonlDataset(path, img_dir, tokenizer, img_transforms, labels, wandb_config.max_seq_length -
                           wandb_config.num_image_embeds - 2)

    return dataset

def evaluate(args, model, tokenizer, data_path, img_dir):
    
    eval_dataset = load_examples(tokenizer, args, data_path, img_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    scores = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            input_ids = batch[0]
            input_modal = batch[2]
            attention_mask = batch[1]
            modal_start_tokens = batch[3]
            modal_end_tokens = batch[4]
            
            if args.multiclass:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    return_dict=True
                )
            else:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=labels,
                    return_dict=True
                )
            logits = outputs.logits
            scores.append(logits.tolist()[0])
            if args.multiclass:
                criterion = get_multiclass_criterion(eval_dataset)
                tmp_eval_loss = criterion(logits, labels)
            else:
                tmp_eval_loss = outputs.loss
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # Move logits and labels to CPU
        if args.multiclass:
            pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
        else:            
            pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy()
        out_label_id = labels.detach().cpu().numpy()
        preds.append(pred)
        out_label_ids.append(out_label_id)

    eval_loss = eval_loss / nb_eval_steps

    result = {"loss": eval_loss}
    # print(preds)

    if args.multiclass:
        tgts = np.vstack(out_label_ids)
        preds = np.vstack(preds)
        result["macro_f1"] = f1_score(tgts, preds, average="macro")
        result["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        preds = [l for sl in preds for l in sl]
        out_label_ids = [l for sl in out_label_ids for l in sl]
        result["accuracy"] = accuracy_score(out_label_ids, preds)
    metrics = f"Test: Accuracy: {(100*result.get('accuracy')):>0.2f}%, Avg loss: {result.get('loss'):>5f} \n"
    return metrics, preds, out_label_ids, scores

def predict_mm(model_path,res_path,metric_path,data_path,typ='predict'):

    parser = argparse.ArgumentParser(f'Project Hyperparameters and Other Configurations Argument Parser')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="bert-base-uncased", type=str, help="Pretrained config name if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--eval_batch_size", default=1, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_seq_length",
        default=300,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_image_embeds", default=3, type=int, help="Number of Image Embeddings from the Image Encoder"
    )

    args = parser.parse_args("")

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = device

    # for multiclass labeling
    args.multiclass = False
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if 
            args.tokenizer_name else args.model_name,
            do_lower_case=True,
            cache_dir=None,
        )

    labels = [0,1]
    num_labels = len(labels)
    transformer_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name, num_labels=num_labels)
    transformer = AutoModel.from_pretrained(args.model_name, config=transformer_config, cache_dir=None)
    img_encoder = ImageEncoderDenseNet(num_image_embeds=args.num_image_embeds)
    multimodal_config = MMBTConfig(transformer, img_encoder, num_labels=num_labels, modal_hidden_size=1024)

    # model = MMBTForClassification(transformer_config, multimodal_config)
    # checkpoint = os.path.join(checkpoint, 'pytorch_model.bin')# ckpt位置
    # fedmodel = FedLightModule.load_from_checkpoint(model_path)
    # model.load_state_dict(fedmodel.model)
    model = FedLightModule.load_from_checkpoint(model_path).model
    model.to(args.device)

    img_dir = os.path.join(data_path,'images')
    text_path = os.path.join(data_path,'meta.jsonl') 
    result, pred_labels, true_labels, scores = evaluate(args, model, tokenizer, text_path, img_dir)
    
    l = len(pred_labels)
    if typ == 'test':
        with open(metric_path, 'w') as f:
            f.write(f'{result}\n')
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

def predict_mm_one_sample(model_path,img_base64, text,device = 'cpu'):

    parser = argparse.ArgumentParser(f'Project Hyperparameters and Other Configurations Argument Parser')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="bert-base-uncased", type=str, help="Pretrained config name if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--eval_batch_size", default=1, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_seq_length",
        default=300,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_image_embeds", default=3, type=int, help="Number of Image Embeddings from the Image Encoder"
    )

    args = parser.parse_args("")

    # Setup CUDA, GPU & distributed training
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = device

    # for multiclass labeling
    args.multiclass = False
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if 
            args.tokenizer_name else args.model_name,
            do_lower_case=True,
            cache_dir=None,
        )

    labels = [0,1]
    num_labels = len(labels)
    transformer_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name, num_labels=num_labels)
    transformer = AutoModel.from_pretrained(args.model_name, config=transformer_config, cache_dir=None)
    img_encoder = ImageEncoderDenseNet(num_image_embeds=args.num_image_embeds)
    multimodal_config = MMBTConfig(transformer, img_encoder, num_labels=num_labels, modal_hidden_size=1024)

    # model = MMBTForClassification(transformer_config, multimodal_config)
    # checkpoint = os.path.join(checkpoint, 'pytorch_model.bin')# ckpt位置
    model = FedLightModule.load_from_checkpoint(model_path).model
    # model.load_state_dict(fedmodel.model)
    model.to(args.device)

    sentence = torch.LongTensor(tokenizer.encode(text, add_special_tokens=True))
    start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
    sentence = sentence[:args.max_seq_length]

    img = base64_pil(img_base64)
    transforms = get_image_transforms()
    image = transforms(img)
    label = torch.LongTensor(np.zeros(1))
    data = {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }
    # eval_sampler = SequentialSampler(dataset)
    # eval_dataloader = DataLoader(
    #     dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    # )
    lens = [len(data["sentence"])]
    bsz, max_seq_len = 1, max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    text_tensor[0, :lens[0]] = data["sentence"]
    mask_tensor[0, :lens[0]] = 1

    img_tensor = torch.stack([data["image"]])
    tgt_tensor = torch.stack([data["label"]])
    img_start_token = torch.stack([data["image_start_token"]])
    img_end_token = torch.stack([data["image_end_token"]])

    # text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor
    model.eval()
    with torch.no_grad():
        # data = tuple(data.to(device))
        labels = tgt_tensor
        input_ids = text_tensor
        input_modal = img_tensor
        attention_mask = mask_tensor
        modal_start_tokens = img_start_token
        modal_end_tokens = img_end_token
        if args.multiclass:
            outputs = model(
                input_modal,
                input_ids=input_ids,
                modal_start_tokens=modal_start_tokens,
                modal_end_tokens=modal_end_tokens,
                attention_mask=attention_mask,
                token_type_ids=None,
                modal_token_type_ids=None,
                position_ids=None,
                modal_position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                return_dict=True
            )
        else:
            outputs = model(
                input_modal,
                input_ids=input_ids,
                modal_start_tokens=modal_start_tokens,
                modal_end_tokens=modal_end_tokens,
                attention_mask=attention_mask,
                token_type_ids=None,
                modal_token_type_ids=None,
                position_ids=None,
                modal_position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=labels,
                return_dict=True
            )
        logits = outputs.logits
        tmp_eval_loss = outputs.loss
        pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy()
        print(pred[0])
        return str(pred[0])

def file_to_base64(path_file):    
    with open(path_file,'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf8')
        return image_base64


if __name__ == "__main__":
    model_path = '/data/projects/fate/examples/gwork/lung/models/model.ckpt'
    # data_path = '/data/projects/fate/examples/gwork/lung/test'
    # img_dir = '/home/qi/guoyujian/working/Multimodal-BERT-in-Medical-Image-and-Text-Classification/data/NLMCXR_png_frontal'
    # res_path = './res.csv'
    # metric_path = './metric.txt'
    # predict_mm(model_path=model_path,res_path = res_path,metric_path=metric_path,data_path=data_path,typ='test')

    img_path = '/data/projects/fate/examples/gwork/lung/train1/images/CXR2644_IM-1130-1001.png'
    text = ' Mild cardiomegaly. Pulmonary vasculature is within normal limits. Costophrenic XXXX are XXXX. There is increased kyphotic curvature of the thoracic spine. Within the heart XXXX, there is a small area of oval-shaped density measuring 2.2 x 1.6 cm without correction for magnification. There is a calcified lymph node in the right hilum. No pneumothorax.'
    img_base64 = file_to_base64(img_path)

    predict_mm_one_sample(model_path = model_path, text = text, img_base64 = img_base64)
