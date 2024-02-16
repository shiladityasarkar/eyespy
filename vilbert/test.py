# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForVisualReasoning
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

import tensorflow as tf
# from tensorflow.python.keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet50
# from tensorflow.python.keras.applications.resnet import preprocess_input
from keras.applications.resnet import preprocess_input
# from tensorflow.keras.preprocessing import image
# from keras.preprocessing import image
import keras.utils as image
# from tensorflow.keras.layers import Dense
from keras.layers import Dense
import numpy as np

# Load pre-trained ResNet50 model without the classification head
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# Define image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Define get_visual_embeddings function
def get_visual_embeddings(image_path, visual_seq_length, visual_embedding_dim):
    img_array = preprocess_image(image_path)
    visual_embeddings = base_model.predict(img_array)

    # Reshape visual embeddings to match required format
    visual_embeddings = np.expand_dims(visual_embeddings, axis=1)  # Add sequence length dimension
    visual_embeddings = np.repeat(visual_embeddings, visual_seq_length, axis=1)  # Repeat for sequence length
    visual_embeddings = np.repeat(visual_embeddings, visual_embedding_dim, axis=2)  # Repeat for embedding dim

    # Convert NumPy array to torch.FloatTensor
    visual_embeddings = torch.tensor(visual_embeddings, dtype=torch.float32)
    return visual_embeddings

text = "What do you see?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings('D:\\wallpapers\\alanwake4.jpg', 10, 512).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor(1).unsqueeze(0)
print(inputs.shape)

# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# scores = outputs.logits