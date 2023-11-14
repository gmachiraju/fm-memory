from typing import List
from itertools import product
import itertools as it
from io import BytesIO
import copy
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns

import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL
from PIL import Image
import cv2
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import scipy

from utils import COCOParser, get_coco_labels, serialize, deserialize, serialize_json, deserialize_json
import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"
SM_CUDA = True

#==================
# Guess 'n' Check
#==================
def GnC(model, processor, img_path, annot_path, text, cache_path, topk=200):
    pass

#==========
# TAGGING
#==========
def similarity_cosine(img_feat, text_feats):
    with torch.no_grad():
        return F.cosine_similarity(img_feat, text_feats)

def tag_inference(model, processor, image, text, chunk_size=10):
    """
    Tagger inference for huggingface CLIP
    """
    model.to(device)
    model.eval()
    if device == "cuda":
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image).to(device)
    
    # prepare images
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    # cosine similarity - chunked for memory efficiency
    sim = []
    matrix_len = len(text)
    for chunk_start in range(0, matrix_len, chunk_size):
        chunk_end = chunk_start + chunk_size
        if chunk_end > matrix_len:
            chunk_end = matrix_len
        text_chunk = text[chunk_start:chunk_end]
        inputs = processor(text_chunk, padding=True, return_tensors="pt")
        text_features = model.get_text_features(**inputs)
        cosine_similarity_chunk = similarity_cosine(image_features, text_features)
        sim.extend(list(cosine_similarity_chunk.cpu().detach().numpy()))
    return torch.from_numpy(np.array(sim)).squeeze()

#================
# SEGMENTATION
#================
def seg_heuristics_dataset(model, processor, img_path, text_list, cache_path):
    files = os.listdir(img_path)
    print("#files to process:", len(files))
    print("#files completed:", len(os.listdir(cache_path)))

    for i, filestr in enumerate(files):
        id_num = filestr.split(".")[0]
        filecache = os.path.join(cache_path, id_num+".json")
        if os.path.isfile(filecache):
            print("JSON exists for: " + id_num + ". Skipping...")
            continue
        else:
            print("Running inference on: " + id_num)
        filename = os.path.join(img_path, filestr)
        image = Image.open(filename)
        region_dict, validity_dict = seg_inference_heuristics(model, processor, image, text_list)
        serialize_json(region_dict, filecache)

def seg_inference_heuristics(model, processor, image, text_list, min_pixels=50, viz_flag=False, verbosity_flag=False):
    """
    Main function to run a segmentation model with vision-based filtration and heuristics
    Input:
        model:
        processor:
        image:
        text_list: 
    output:
    """
    id2query_dict, validity_dict = {}, {}
    prev_size = 0
    for i, query in enumerate(text_list):
        mask = seg_inference_query(model, processor, image, query) # seg inference
        if i == 0:
            aggregated_mask = np.zeros_like(mask)
        exit_code, obj_mask = label_filtration_vision(mask, min_pixels=min_pixels) # CC analysis 
        curr_size = np.sum(obj_mask)
        validity_dict[query] = exit_code
        if exit_code == 1:
            aggregated_mask += (obj_mask * (i+1))
            if prev_size > curr_size:
                level_set = i+1
            else:
                level_set = i
            aggregated_mask = np.where(aggregated_mask > (i+1), level_set, aggregated_mask) # ceiling
        id2query_dict[i+1] = query
        prev_size = 0.0 + curr_size # update
        if verbosity_flag == True:
            print(i+1, query, exit_code)
            print("-"*20)

    # update text_list: filtered for any queries without sufficient evidence 
    text_set = set([key for key in validity_dict.keys() if validity_dict[key] == 1])
    # create bounding box for the regions
    all_hits = np.where(aggregated_mask > 0, 1, 0) # reset any positive values to 1   
    if viz_flag == True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax.imshow(aggregated_mask)
    
    gray = np.array(all_hits*255).astype('uint8')
    thresh = cv2.threshold(gray, 128,255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    region_dict = {}
    num_skips = 0
    for i, coords in enumerate(contours):
        x,y,w,h = cv2.boundingRect(coords)
        if (w * h) < min_pixels:
            num_skips += 1
            continue
        crop = aggregated_mask[y:y+h, x:x+w]
        if viz_flag == True:
            rect = plt.Rectangle((x, y), w, h, linewidth=4, edgecolor="white", facecolor='none')
            ax.add_patch(rect)
        ids = list(set(list(crop.flatten())))
        tags = [id2query_dict[id_num] for id_num in ids if id_num > 0]
        tags = [tag for tag in tags if tag in text_set] # refine
        region_dict[i+1-num_skips] = {'bbox': [x,y,w,h], "polygon": coords.tolist(), "tags": tags}
    
    region_dict["all_labels"] = list(text_set) # save all labels that are predicted
    region_dict["method"] = "direct clipseg heuristics"
    if viz_flag == True:
        plt.show()
    return region_dict, validity_dict

def seg_inference_query(model, processor, image, query):
    torch.cuda.empty_cache()
    if SM_CUDA == False and device == "cpu":
        pass
    elif device == "cuda":
        print("GPU inference initiated...")
    model.to(device)
    model.eval()
    
    # Now we get chunked output into a mask
    if device == "cuda":
        transform = transforms.Compose([transforms.ToTensor()])
        image_pt = transform(image).to(device)
        inputs = processor(text=[query], images=[image_pt], padding="max_length", return_tensors="pt")
    else:
        inputs = processor(text=[query], images=[image], padding="max_length", return_tensors="pt")
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)
    
    outputs = model(**inputs)
    pred = outputs.logits.unsqueeze(0).unsqueeze(0)
    mask = cv2.resize(torch.sigmoid(pred[0,0,:,:].cpu().detach()).numpy(), image.size)
    return mask

def label_filtration_vision(mask, min_pixels=50, verbosity_flag=False):
    """
    Connected components analysis & heuristics to filter out labels
    Takes in a single mask and runs analysis to decide on label's relevance
    Input:
        mask: numpy array to analyze
    Output:
        exit_code: 0 means query tossed, 1 means valid
        binary: thresholded mask to show object
    """
    # feature scaling & binarization
    thresh, binary = feature_scale(mask)
    if np.sum(binary) < min_pixels: 
        if verbosity_flag == True:
            print("miniscule item detected")
        exit_code = 0
        binary = np.zeros_like(binary) # zero mask
        return exit_code, binary
    # Apply CC analysis heuristics
    exit_code = run_cc_analysis(thresh, tol_objs=3)
    if exit_code == 0:
        if verbosity_flag == True:
            print("too many CCs detected")
        binary = np.zeros_like(binary)
    else:
        if verbosity_flag == True:
            print("valid object detected")
    return exit_code, binary

def feature_scale(mask):
    """
    Input: 
        mask: probability mask from a segmentation model
    Output: 
        thresh: thresholded mask after otsu [0,255]
        binary: thresholded mask after otsu {0,1}
    """
    m = ((mask - mask.min()) * (1/(mask.max() - mask.min()) * 255)).astype('uint8')
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = np.where(m > ret, 1.0, 0.0)
    return thresh, binary

def run_cc_analysis(thresh, tol_objs=5):
    """
    Runs 8-way connectivity CC analysis
    Input:
        thresh: threhsolded/binarized mask 
        tol_objects: number of tolerated CCs
    Output:
        exit_code: 0 means label is tossed, 1 means label valid
    """
    analysis = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    num_labels = analysis[0]
    if num_labels > tol_objs:
        return 0
    return 1

#======================
# Peliminary analysis 
#======================
def get_tag_statistics(probs, vocab):
    """
    probs: the list of probabilities or cosine similarities from a tagger model
    vocab: the list of labels 
    """
    pairs = zip(vocab, probs.cpu().detach().numpy())
    sorted_pairs = sorted(pairs, reverse=True, key=lambda t: t[1])
    vocab = [t[0] for t in sorted_pairs]
    prob = [t[1] for t in sorted_pairs]
    return sorted_pairs, np.mean(prob), np.std(prob)

def display_bars(probs, vocab, mode="full", dims=None, top_k=10):
    if len(vocab) > 10:
        print("Vocab too large to display all labels... displaying top 10 in print-outs")
        
    if mode == "full":
        fig,ax = plt.subplots(1,2,figsize=(5, 5))
        pairs = zip(vocab, probs.cpu().detach().numpy())
        sorted_pairs = sorted(pairs, reverse=True, key=lambda t: t[1])
        
        # bar chart
        topk = sorted_pairs[:top_k]
        vocabk = [t[0] for t in topk]
        probk = [t[1] for t in topk]
        for idx,p in enumerate(probk):
            print(vocabk[idx],"--", p.item())
        y_pos = np.arange(len(vocabk))
        ax[0].barh(y_pos[::1], probk[::-1], align='center')
        ax[0].set_yticks(y_pos[::1], labels=vocabk[::-1])

        # histogram
        vocab = [t[0] for t in sorted_pairs]
        prob = [t[1] for t in sorted_pairs]
        ax[1].hist(prob, bins=len(prob)//4)
        plt.show()

def display_image(image, mode="full"):
    """
    Uses the PIL library to viz an image
    image: PIL image object
    mode: "full" refers to full image not partitioned into chunks
          "chunks" refers to a grid of chunked images (deprecated)
    """
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

def tag_inference_coco(model, processor, img_path, annot_path, text, cache_path, version="hf", topk=200):
    """
    Tagger ran on COCO-like dataset. Computes statistics in the loop. 
    NOTE: We should save outputs and compute statistics later. May switch to other metrics.
    """
    coco = COCOParser(annot_path, img_path)
    files = os.listdir(img_path)
    print("#files to process:", len(files))

    if os.path.isfile(os.path.join(cache_path,"probs-labels-"+str(len(text))+".obj")):
        print("prior cache found!")
        topk_probs = deserialize(os.path.join(cache_path,"probs-labels-"+str(len(text))+".obj"))
        stats = deserialize(os.path.join(cache_path,"stats-labels-"+str(len(text))+".obj"))
        print(len(topk_probs.keys()), "images processed")
    else:
        print("no prior cache found")
        topk_probs = {}
        stats = {}

    for i, filestr in enumerate(files):
        if (i+1) % 20 == 0:
            print("Processed", (i+1), "files")
        if (filestr in topk_probs.keys()) and (filestr in stats.keys()):
            continue
        filename = os.path.join(img_path, filestr)
        image = Image.open(filename)
        # tagger inference
        probs = tag_inference(model, processor, image, text, version=version)        
        # distribution summary stats
        _, mu, sigma = get_tag_statistics(probs, text)

        preds = dict(zip(text, list(probs.detach().numpy())))
        # sort by probability + save topk
        sorted_label_preds = sorted(preds.items(), reverse=True, key=lambda item: item[1])
        if len(text) < topk:
            topk_probs[filestr] = sorted_label_preds
        else:
            topk_probs[filestr] = sorted_label_preds[:topk]

        counts, sizes = get_coco_labels(coco, img_path, filestr)
        # check to see if an image has fewer than 2 objects -- the skip
        count_sum = np.sum(np.array(list(counts.values())))
        if count_sum < 2:
            continue
        scaled = {}
        for k in counts.keys():
            if sizes[k] > 0 and counts[k] > 0:
                scaled[k] = sizes[k] / counts[k] 
            else:
                scaled[k] = 0
        multihot = dict(zip([key for key in counts.keys()], [int(val > 0) for val in counts.values()]))
        
        # pad labels and trim predictions to get top50 objects for NDCG
        top50_preds = dict(sorted_label_preds[:50])
        counts = pad_label_dict(top50_preds,counts)
        sizes = pad_label_dict(top50_preds,sizes)
        scaled = pad_label_dict(top50_preds,scaled)
        multihot = pad_label_dict(top50_preds,multihot)

        # sorted_preds = list(dict(sorted(preds.items())).values())
        top50_preds = list(dict(sorted(top50_preds.items())).values())
        sorted_counts = list(dict(sorted(counts.items())).values())
        sorted_sizes = list(dict(sorted(sizes.items())).values())
        sorted_scaled = list(dict(sorted(scaled.items())).values())
        sorted_multihot = list(dict(sorted(multihot.items())).values())

        ndcg_count = ndcg_score([sorted_counts], [top50_preds])
        ndcg_size = ndcg_score([sorted_sizes], [top50_preds])
        ndcg_scaled = ndcg_score([sorted_scaled], [top50_preds])
        ndcg_hot = ndcg_score([sorted_multihot], [top50_preds])
        pearson = scipy.stats.pearsonr(sorted_sizes, top50_preds)[0]
        spearman = scipy.stats.spearmanr(sorted_sizes, top50_preds)[0]
        kendall = scipy.stats.kendalltau(sorted_sizes, top50_preds)[0]
        stats[filestr] = {"mu": mu, "sigma": sigma, 
                          "ndcg_count": ndcg_count, "ndcg_size": ndcg_size,
                          "ndcg_scaled": ndcg_scaled, "ndcg_hot": ndcg_hot,
                          "pearson": pearson, "spearman": spearman, "kendall": kendall}
        # save
        serialize(stats, os.path.join(cache_path,"stats-labels-"+str(len(text))+".obj"))
        serialize(topk_probs, os.path.join(cache_path,"probs-labels-"+str(len(text))+".obj"))
        
        # small sample for now
        if i > 500:
            break
    return 
        
def pad_label_dict(preds, labels):
    """
    helper function to pad labels
    """
    new_labels = {}
    for key in preds.keys():
        if key not in labels.keys():
            new_labels[key] = 0
        else:
            new_labels[key] = labels[key]
    return new_labels

def tag_analyze_inference_dataset(stats_labels1, stats_labels2, titlestr="CLIP tagging performance on COCO"):
    """
    Performs manipulation of results to eventually pass to a plotting function.
    """
    mu1 = [stats_labels1[key]["mu"] for key in stats_labels1.keys()]
    sigma1 = [stats_labels1[key]["sigma"] for key in stats_labels1.keys()]
    nc1 = [stats_labels1[key]["ndcg_count"] for key in stats_labels1.keys()]
    ns1 = [stats_labels1[key]["ndcg_size"] for key in stats_labels1.keys()]
    nh1 = [stats_labels1[key]["ndcg_hot"] for key in stats_labels1.keys()]
    p1 = [stats_labels1[key]["pearson"] for key in stats_labels1.keys()]

    mu2 = [stats_labels2[key]["mu"] for key in stats_labels2.keys()]
    sigma2 = [stats_labels2[key]["sigma"] for key in stats_labels2.keys()]
    nc2 = [stats_labels2[key]["ndcg_count"] for key in stats_labels2.keys()]
    ns2 = [stats_labels2[key]["ndcg_size"] for key in stats_labels2.keys()]
    nh2 = [stats_labels2[key]["ndcg_hot"] for key in stats_labels2.keys()]
    p2 = [stats_labels2[key]["pearson"] for key in stats_labels2.keys()]
    
    # pad if results are not on same number of samples
    labels1_samples = len(stats_labels1.keys())
    labels2_samples = len(stats_labels2.keys())
    min_samples = np.min([labels1_samples, labels2_samples])
    if labels1_samples > labels2_samples:
        mu1 = mu1[:min_samples]
        sigma1 = sigma1[:min_samples]
        nc1 = nc1[:min_samples]
        ns1 = ns1[:min_samples]
        nh1 = nh1[:min_samples]
        p1 = p1[:min_samples]
    
    stats_dict = {"mu1": mu1, "sigma1": sigma1, "ndcg_count1": nc1, 
                  "ndcg_sizes1": ns1, "ndcg_hot1": nh1, "pearson1": p1,
                  "mu2": mu2, "sigma2": sigma2, "ndcg_count2": nc2, 
                  "ndcg_sizes2": ns2, "ndcg_hot2": nh2, "pearson2": p2}
    df = pd.DataFrame.from_dict(stats_dict)
    df["id"] = df.index

    # Create a new DataFrame for grouped bar plot
    melted_df = pd.melt(df, id_vars=["id"], value_vars=df.columns, var_name='Variable', value_name='Value')
    melted_df["vocab"] = [var[-1] for var in melted_df["Variable"]]
    melted_df["var"] = [var[:-1] for var in melted_df["Variable"]]
    
    plt.figure()
    g = sns.boxplot(melted_df, x="var", y="Value", hue="vocab", legend=True)
    # plt.legend(title='Vocabulary', labels=['COCO','COCO+IN'])
    leg = g.axes.get_legend()
    new_title = 'Vocabulary'
    leg.set_title(new_title)
    new_labels = ['COCO', 'COCO+IN']
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    
    plt.xticks(rotation=30, ticks=[0,1,2,3,4,5], labels=["Mean(P)", "Std Dev(P)", "Count NDCG", "Size NDCG", "Hot NDCG", "Pearson"],  horizontalalignment='right')
    plt.tight_layout()
    plt.ylabel("")
    plt.xlabel("")
    plt.title(titlestr)
    plt.show()

def seg_all_outputs(model, processor, image, text_labels):
    """
    Used for error analysis of segmentation models.
    """
    if type(text_labels) == dict:
        text_list = list(text_labels.keys())
    else:
        text_list = text_labels
    mask = seg_inference_vocab(model, processor, image, text_list, sample_flag=False)
    return mask

def seg_inference_vocab(model, processor, image, text_list, sample_flag=False, chunk_size=5):
    """
    Inference method for SMs like CLIPSeg
    """
    torch.cuda.empty_cache()
    if SM_CUDA == False and device == "cpu":
        pass
    else:
        print("Running with GPU inference")
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
    if num_prompts > chunk_size:
        if device == "cuda":
            print("Warning: too many prompts for GPU memory, batching with chunks of size", chunk_size, "...")
            # chunked_text_list = [list(a) for a in np.array_split(text_list, chunk_size)]
            chunked_text_list = [text_list[x:x+chunk_size] for x in range(0, num_prompts, chunk_size)]
        else:
            print("Batching with chunks of size", chunk_size, "...")
            # chunked_text_list = [list(a) for a in np.array_split(text_list, chunk_size)]
            chunked_text_list = [text_list[x:x+chunk_size] for x in range(0, num_prompts, chunk_size)]
    else:
        chunked_text_list = [text_list]
    num_chunks = len(chunked_text_list)
    
    # Now we get chunked output into a mask
    mask = np.zeros(image.size).T
    for ctl in chunked_text_list:
        batch_size = len(ctl)
        if device == "cuda":
            transform = transforms.Compose([transforms.ToTensor()])
            image_pt = transform(image).to(device)
            inputs = processor(text=ctl, images=[image_pt] * batch_size, padding="max_length", return_tensors="pt")
        else:
            inputs = processor(text=ctl, images=[image] * batch_size, padding="max_length", return_tensors="pt")
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)

        if chunk_size == 1:
            pred = outputs.logits.unsqueeze(0).unsqueeze(0)
        else:
            pred = outputs.logits.unsqueeze(1)
    
        for i in range(batch_size):
            mask += cv2.resize(torch.sigmoid(pred[i,0,:,:].cpu().detach()).numpy(), image.size)
    return mask

def viz_seg_mask(mask, mode="probs"):
    plt.figure()
    if mode == "probs":
        cmap = "jet"
        im = plt.imshow(mask, cmap=cmap)
    elif mode == "discrete":
        cmap = "Set2"
        im = plt.imshow(mask, cmap=cmap)
    plt.axis("off")
    if mode == "probs":
        plt.colorbar(orientation="horizontal", pad=0.03)
        plt.show()
    elif mode == "discrete":
        # mat = plt.matshow(mask, cmap=cmap, vmin=np.min(mask) - 0.5, vmax=np.max(mask) + 0.5)
        cax = plt.colorbar(im, orientation="horizontal", ticks=np.arange(np.min(mask), np.max(mask) + 1), pad=0.03)
        # plt.colorbar()
        plt.show()


def main(): 
    # Currently, this code runs the COCO tagging analysis (box/whisker plots)
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    from utils import COCOParser
    coco_annotations_file = "/oak/stanford/groups/paragm/gautam/gcp_backup/data/coco/trainval_annotations/instances_val2017.json"
    coco_images_dir = "/oak/stanford/groups/paragm/gautam/gcp_backup/data/coco/val2017/val2017"
    coco = COCOParser(coco_annotations_file, coco_images_dir)

    from utils import process_coco_labels
    vocab_data_path = "/oak/stanford/groups/paragm/gautam/gcp_backup/ibm-work/data/coco_objects/coco-labels-2014_2017.txt"
    vocab_data = process_coco_labels(vocab_data_path)

    from utils import process_imagenet_labels
    vocab_nondata_path = "/oak/stanford/groups/paragm/gautam/gcp_backup/data/imagenet/imagenet1000_clsidx_to_labels.txt" 
    vocab_nondata = process_imagenet_labels(vocab_nondata_path)
    vocab_extra = list(set(vocab_data + vocab_nondata))
    print("vocab size:", len(vocab_extra))

    cache_path = "/oak/stanford/groups/paragm/gautam_development/ibm-work/fm-memory/src/outputs"
    tag_inference_coco(clip, clip_processor, coco_images_dir, coco_annotations_file, vocab_extra, cache_path)

if __name__ == "__main__":
    main()
