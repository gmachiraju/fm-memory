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
import random

import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"
SM_CUDA = False


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

def vlm_inference(model, processor, image, text, version="hf"):
    model.to(device)
    model.eval()
    if device == "cuda":
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).to(device)
    
    if version == "oa": # OpenAI
        # img_inputs = processor(images=image, return_tensors="pt", padding=True)
        # img_inputs['pixel_values'] = img_inputs['pixel_values'].to(device)
        # embeddings = model.get_image_features(**img_inputs)
        Exception("Error: OpenAI variant not implemented")
    elif version == "hf": # huggingface
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs

def vlm_inference_chunks(model, processor, image, text):
    chunks, h, w = tile(image)    
    probs_vec = []
    for chunk in chunks:
        probs = vlm_inference(model, processor, chunk, text)
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

def sm_inference(model, processor, image, text_list, sample_flag=False):
    """
    Inference method for SMs like CLIPSeg
    """
    torch.cuda.empty_cache()
    if SM_CUDA == False:
        device = "cpu"
        print("Overwriting to CPU run on SM inference...")
    model.to(device)
    model.eval()

    # if sample_flag, then we sample a portion of the text_list
    if sample_flag == True:
        print("sample_flag is True, sampling a proportion of prompts...")
        max_prompts = 20
        num_prompts = len(text_list)
        to_sample = min(max_prompts, num_prompts)
        text_list = random.sample(text_list, to_sample)
        
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

def sm_filter_by_probs(threshold, text_dict):
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
    return text_list, num_prompts

def run_cc_analysis(colored, thresh, tol_objs):
    filtered, idxs_kept = [],[]
    for idx, c in enumerate(colored):
        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(thresh[idx], 8, cv2.CV_32S)
        num_labels = analysis[0]
        if num_labels > tol_objs:
            c = np.zeros_like(c)
        else:
            idxs_kept.append(idx)
        filtered.append(c)
    return filtered, idxs_kept

def filter_by_connected_components(image, masks, num_prompts, text_list, cmap, viz_flag=False):
    if viz_flag == True:
        _, ax = plt.subplots(1, num_prompts+1, figsize=(15, num_prompts))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)

    colored, thresh = [], []
    vmax = len(masks)
    for idx, m in enumerate(masks):
        m = ((m - m.min()) * (1/(m.max() - m.min()) * 255)).astype('uint8')
        ret,thr = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        m = np.where(m > ret, idx+1, 0.0)
        thresh.append(thr)
        colored.append(m)
        if viz_flag == True:
            ax[idx+1].imshow(m, cmap=cmap, vmin=0.001, vmax=vmax)
            ax[idx+1].text(0, -15, text_list[idx])
    
    # Now we filter
    if viz_flag == True:
        _, ax = plt.subplots(1, num_prompts+1, figsize=(15, num_prompts))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)

    # now we adaptively apply heuristics
    filtered, idxs_kept = run_cc_analysis(colored, thresh, 2)
    if idxs_kept == []:
        print("Relaxing tolerance to more connected components")
        filtered, idxs_kept = run_cc_analysis(colored, thresh, 3) # up the tolerance
    # maybe switch to while loop

    if viz_flag == True:
        for idx,c in enumerate(filtered):
            ax[idx+1].imshow(c, cmap=cmap, vmin=0.001, vmax=vmax)
            ax[idx+1].text(0, -15, text_list[idx])

    return filtered, idxs_kept

def sm_aggregate(model, processor, image, text_labels, viz_flag=False, threshold=0.5, filter_flag=False):
    """
    This function performs inference and creates an aggregated mask over all viable labels
    Viability is determined based on proabbility threshold
    """
    if filter_flag == True:
        assert type(text_labels) == dict
        sm_filter_by_probs(threshold, text_dict)
    else:
        if type(text_labels) == dict:
            text_list = list(text_labels.keys())
        else:
            text_list = text_labels
        num_prompts = len(text_list)

    masks = sm_inference(model, processor, image, text_list, sample_flag=False)

    # visualize prediction
    #-----------------------
    my_cmap = plt.get_cmap('Accent')
    my_cmap.set_under('black')
    
    # all prompts 
    if viz_flag == True:
        _, ax = plt.subplots(1, num_prompts+1, figsize=(15, num_prompts))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
        [ax[i+1].imshow(masks[i]) for i in range(num_prompts)]
        [ax[i+1].text(0, -15, text_list[i]) for i in range(num_prompts)]

    # filter by connected components
    filtered_masks, idxs_kept = filter_by_connected_components(image, masks, num_prompts, text_list, my_cmap, viz_flag=viz_flag)

    # aggregate
    if viz_flag == True:
        _, ax = plt.subplots(1, 3, figsize=(15, 15))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)

    full_mask = np.zeros(masks[0].shape)
    for idx, m in enumerate(masks):
        if idx in idxs_kept:
            full_mask += m
    
    leveled_mask = np.zeros(masks[0].shape)
    for idx, m in enumerate(filtered_masks):
        leveled_mask += m

    if viz_flag == True:
        ax[1].imshow(full_mask)
        ax[2].imshow(leveled_mask)

    return [filtered_masks[idx] for idx in idxs_kept]

def sm_hallucination_viz(image, masks, text_dict):
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

def bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def hyfive(sm_model, sm_processor, vlm_model, vlm_processor, image, vocab):
    # 1. Run segmentation model (SM) to get hypothetical object locations
    hypotheses = sm_aggregate(sm_model, sm_processor, image, vocab, viz_flag=True, filter_flag=False)
    # 2. Get bounding boxes for hypotheses
    bbs = []
    for h in hypotheses:
        bbs.append(bbox(h))

    # 3. Run vision-language model (VLM) on each hypothesis region
    # for bb in bbs:
    #     patch = np.asarray(image)[bb] 
    #     patch = PIL.Image.fromarray(numpy.uint8(patch))
    #     probs = vlm_inference(vlm_model, vlm_processor, patch, vocab)
    
    # use Probabilities to then filter best hits per hypothesis and then extract them

    # sm_inference(model, processor, image, text_list, sample_flag=False)