from typing import List
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from io import BytesIO
import copy

import torch
from torchvision import transforms
import numpy as np
import PIL
from PIL import Image
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2

import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIPSEG_CUDA = False

def tile(img, d=224):
    """
    img: PIL.Image
    """
    patches = []
    w, h = img.size
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        patches.append(np.array(img.crop(box)))
    return patches, w // d, h // d


def process_chunks(image):
    chunks, h, w = tile(image)  
    return chunks


def clip_inference(model, processor, image, text, version="hf"):
    model.to(device)
    model.eval()

    if device == "cuda":
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).to(device)
    
    if version == "oa":
        # img_inputs = processor(images=image, return_tensors="pt", padding=True)
        # img_inputs['pixel_values'] = img_inputs['pixel_values'].to(device)
        # embeddings = model.get_image_features(**img_inputs)
        Exception("Error: OpenAI variant not implemented")

    elif version == "hf":
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs


def clip_inference_chunks(model, processor, image, text):
    chunks, h, w = tile(image)    
    probs_vec = []
    for chunk in chunks:
        probs = clip_inference(model, processor, chunk, text)
        probs_vec.append(probs)
    return probs_vec


def display_bars(probs, vocab, mode="full", dims=None):
    if len(vocab) > 10:
        print("Vocab too large to display all labels... displaying top 10")
        
    if mode == "full":
        fig = plt.figure(figsize=(2, 2))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 1),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for ax, prob in zip(grid, probs):
            pairs = zip(vocab, prob.cpu().detach().numpy())
            top10 = sorted(pairs, reverse=True, key=lambda t: t[1])[:10]
            vocab = [t[0] for t in top10]
            prob = [t[1] for t in top10]

            for idx,p in enumerate(prob):
                print(vocab[idx],"--", p.item())

            y_pos = np.arange(len(vocab))
            bars = list(prob)
            ax.barh(y_pos, bars, align='center')
            ax.set_yticks(y_pos, labels=vocab)
        plt.show()

    elif mode == "chunks":
        h,w = dims
        fig = plt.figure(figsize=(2, 30))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(w, h),  # creates 2x2 grid of axes
                        axes_pad=0.15,  # pad between axes in inch.
                        )

        # first get high-prob labels across chunks
        label_dict = {}
        for v in vocab:
            label_dict[v] = 0.0
        for ax, prob in zip(grid, probs):
            # pdb.set_trace()
            pairs = list(zip(vocab, list(prob.cpu().detach().numpy()[0])))
            for p in pairs:
                label_dict[p[0]] = np.max([label_dict[p[0]], p[1]])

        # next we pick the top labels across the chunks
        pairs = list(label_dict.items())
        top10 = sorted(pairs, reverse=True, key=lambda t: t[1])[:10]
        top10_vocab = [t[0] for t in top10]

        # then we plot
        for ax, prob in zip(grid, probs):
            pairs = dict(zip(vocab, list(prob.cpu().detach().numpy()[0])))
            bars = [pairs[v] for v in top10_vocab]

            y_pos = np.arange(len(top10_vocab))
            ax.barh(y_pos, bars, align='center')
            ax.set_yticks(y_pos, labels=top10_vocab)
            ax.set_xlim(0,1)

        plt.show()


def display_image(image, mode="full"):
    if mode == "full":
        image.show()
        return None

    elif mode == "chunks":
        chunks, h, w = tile(image)  
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(w, h),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for ax, im in zip(grid, chunks):
            ax.imshow(im)
        plt.show()
        return h,w
        

def extract_entities(probs_whole, probs_chunk, vocab):
    entities_dict = {}
    
    for idx, prob in enumerate(list(probs_whole.flatten())):
        term = vocab[idx]
        if term not in entities_dict.keys():
            entities_dict[term] = prob.item()

    entities_dict_whole = copy.deepcopy(entities_dict)
    entities_dict_chunk = {}

    for prob_vec in probs_chunk:
        for idx, prob in enumerate(list(prob_vec.flatten())):
            term = vocab[idx]
            # chunk handling
            if term not in entities_dict_chunk.keys():
                entities_dict_chunk[term] = prob.item() 
            else:
                max_prob = np.max([entities_dict_chunk[term], prob.item()]) 
                entities_dict_chunk[term] = max_prob
           
            max_prob = np.max([entities_dict[term], prob.item()]) 
            entities_dict[term] = max_prob

    return entities_dict, entities_dict_whole, entities_dict_chunk


def clipseg_inference(model, processor, image, text_list):
    """
    Inference methof for CLIPSeg
    """
    torch.cuda.empty_cache()
    if CLIPSEG_CUDA == False:
        device = "cpu"
        print("Overwriting to CPU run on CLIPSeg inference...")
    model.to(device)
    model.eval()

    num_prompts = len(text_list)
    chunk_size = 5
    if num_prompts > chunk_size and device == "cuda":
        print("Warning: too many prompts for GPU memory, clipping to chunks of size", chunk_size, "...")
        chunked_text_list = [list(a) for a in np.array_split(text_list, chunk_size)]
    else:
        chunked_text_list = [text_list]
    
    preds = []
    for text_list in chunked_text_list:
        num_prompts = len(text_list) # get the chunked length
        if device == "cuda":
            transform = transforms.Compose([transforms.ToTensor()])
            image_pt = transform(image).to(device)
            inputs = processor(text=text_list, images=[image_pt] * num_prompts, padding="max_length", return_tensors="pt")
        else:
            inputs = processor(text=text_list, images=[image] * num_prompts, padding="max_length", return_tensors="pt")

        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)

        if num_prompts == 1:
            pred = outputs.logits.unsqueeze(0).unsqueeze(0)
        else:
            pred = outputs.logits.unsqueeze(1)
        preds.extend(pred)

    masks = [cv2.resize(torch.sigmoid(preds[i][0].cpu().detach()).numpy(), image.size) for i in range(num_prompts)]
    return masks


def clipseg_aggregate(model, processor, image, text_dict, viz_flag=False, threshold=0.5):
    """
    This function performs inference and creates an aggregated mask over all viable labels
    Viability is determined based on proabbility threshold
    """
    print("Using probability threshold of", threshold)
    text_list = []
    for text in text_dict.keys():
        if text_dict[text] > threshold:
            text_list.append(text)
    print("Label prompts with high probability:", text_list)

    # clipseg inf
    #------------
    num_prompts = len(text_list)
    if num_prompts == 0:
        print("No labels with high probability")
        return None, None
    masks = clipseg_inference(model, processor, image, text_list)

    # visualize prediction
    #-----------------------
    my_cmap = plt.get_cmap('Accent')
    my_cmap.set_under('black')

    # all prompts 
    if viz_flag == True:
        _, ax = plt.subplots(1, num_prompts+1, figsize=(15, num_prompts))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
        
    if viz_flag == True:
        [ax[i+1].imshow(masks[i]) for i in range(num_prompts)]
        [ax[i+1].text(0, -15, text_list[i]) for i in range(num_prompts)]
        
    # colored versions
    if viz_flag == True:
        _, ax = plt.subplots(1, num_prompts+1, figsize=(15, num_prompts))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
    colored = []
    vmax = len(masks)
    for idx, m in enumerate(masks):
        m = ((m - m.min()) * (1/(m.max() - m.min()) * 255)).astype('uint8')
        ret,thr = cv2.threshold(m, 0, 130, cv2.THRESH_OTSU)
        m = np.where(m > ret, idx+1, 0.0)
        colored.append(m)
        if viz_flag == True:
            ax[idx+1].imshow(m, cmap=my_cmap, vmin=0.001, vmax=vmax)
            ax[idx+1].text(0, -15, text_list[idx])

    if viz_flag == True:
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
    full_mask = np.zeros(masks[0].shape)
    for m in masks:
        full_mask += m
    if viz_flag == True:
        ax[1].imshow(full_mask)

    if viz_flag == True:
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
    full_mask = np.zeros(masks[0].shape)
    for m in colored:
        to_add = np.nonzero(m)
        full_mask[to_add] = m[to_add]
    if viz_flag == True:
        ax[1].imshow(full_mask, cmap=my_cmap, vmin=0.001, vmax=vmax)

    return full_mask, vmax


def clipseg_hallucination_viz(image, masks, text_dict):
    text_list = list(text_dict.keys())
    probs = list(text_dict.values())
    num_prompts = len(text_list)

    w = 5
    h = ((num_prompts+1) // w) + 1
    
    fig, ax = plt.subplots(h, w, figsize=(w,h))
    [a.axis('off') for a in ax.flatten()]
    ax[0,0].imshow(image)
    fig.suptitle("Probability Heatmap")
    for i, a in enumerate(ax.flatten()[1:]):
        try:
            a.imshow(masks[i])
            a.text(0, -15, text_list[i])
        except IndexError: # empty axes
            continue

    fig, ax = plt.subplots(h, w, figsize=(w,h))
    [a.axis('off') for a in ax.flatten()]
    ax[0,0].imshow(image)
    fig.suptitle("Probability Heatmap w/ alpha=CLIP's Probability")
    for i, a in enumerate(ax.flatten()[1:]):
        try:
            a.imshow(masks[i], alpha=probs[i])
            a.text(0, -15, text_list[i])
        except IndexError: # empty axes
            continue

    return 



# def sam_inference(entities_dict, predictor, image):
#     predictor.set_image(np.asarray(image))
#     pdb.set_trace()

#     masks, scores, _ = predictor.predict(box=box_array)
#     best_idx = np.argsort(scores)[-1]
#     predictions = [(masks[best_idx], box["category"])]
#     segmentations.extend(predictions)
#     combined_segmentations.extend(segmentations)


# def clipseg_inference_large(model, processor, image, text_dict):
#     """
#     CLIPSeg inference run for a large corpus of labels
#     """
#     text_list = list(text_dict.keys())

#     # clipseg inf
#     #------------
#     num_prompts = len(text_list)
#     masks = clipseg_inference(model, processor, image, text_list)
#     return masks