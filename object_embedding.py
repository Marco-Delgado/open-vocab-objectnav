import clip
import torch
import pickle
import numpy as np

model, preprocess = clip.load("ViT-B/32", device="cpu")
objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa", "couch"]
prompts = ["a photo of " + object_ for object_ in objects]
object_embeddings = {}
for i, prompt in enumerate(prompts):
    tokenized_prompt = clip.tokenize(prompt)
    with torch.no_grad():
        embedding = model.encode_text(tokenized_prompt)

    object_embeddings[objects[i]] = np.array(embedding)[0]

with open('embeddings.pickle', 'wb') as f:
    pickle.dump(object_embeddings, f)
