import os
import pickle
import torch
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet201, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------------------
# PATH CONFIG
# ---------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_model.h5")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_tokenizer.pkl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_metadata.pkl")

ATTN_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer_attention.pkl")
ATTN_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "metadata_attention.pkl")

ENCODER_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "encoder_attention.weights.h5")
DECODER_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "decoder_attention.weights.h5")

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ---------------------------
# CNN FEATURE EXTRACTOR
# ---------------------------

feature_extractor = DenseNet201(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):

    img = load_img(img_path, target_size=(224,224))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img,0)

    return feature_extractor.predict(img, verbose=0)

# ---------------------------
# LOAD CNN-LSTM
# ---------------------------

cnn_model = load_model(CNN_MODEL_PATH, compile=False)

with open(TOKENIZER_PATH,"rb") as f:
    tokenizer = pickle.load(f)

with open(METADATA_PATH,"rb") as f:
    max_length = pickle.load(f)["max_length"]

index_word = {v:k for k,v in tokenizer.word_index.items()}

def generate_cnn_caption(img_path):

    feature = extract_features(img_path)

    text="startseq"

    for _ in range(max_length):

        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], max_length)

        yhat = cnn_model.predict([feature,seq],verbose=0)

        word_id = np.argmax(yhat)
        word = index_word.get(word_id)

        if word is None:
            break

        text+=" "+word

        if word=="endseq":
            break

    return text.replace("startseq","").replace("endseq","").strip()

# ---------------------------
# ATTENTION MODEL
# ---------------------------

EMBED_DIM=256
UNITS=512

class Encoder(tf.keras.Model):

    def __init__(self,embed_dim):

        super().__init__()

        self.cnn = ResNet50(include_top=False,weights="imagenet")
        self.cnn.trainable=False
        self.fc = tf.keras.layers.Dense(embed_dim)

    def call(self,x):

        x=self.cnn(x)
        x=tf.reshape(x,(x.shape[0],-1,x.shape[3]))
        x=self.fc(x)

        return tf.nn.relu(x)

class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self,units):

        super().__init__()

        self.W1=tf.keras.layers.Dense(units)
        self.W2=tf.keras.layers.Dense(units)
        self.V=tf.keras.layers.Dense(1)

    def call(self,features,hidden):

        hidden=tf.expand_dims(hidden,1)

        score=tf.nn.tanh(self.W1(features)+self.W2(hidden))

        attention_weights=tf.nn.softmax(self.V(score),axis=1)

        context_vector=attention_weights*features
        context_vector=tf.reduce_sum(context_vector,axis=1)

        return context_vector,attention_weights

class Decoder(tf.keras.Model):

    def __init__(self,embed_dim,units,vocab_size):

        super().__init__()

        self.units=units

        self.embedding=tf.keras.layers.Embedding(vocab_size,embed_dim)

        self.attention=BahdanauAttention(units)

        self.lstm=tf.keras.layers.LSTM(units,return_sequences=True,return_state=True)

        self.fc1=tf.keras.layers.Dense(units)
        self.fc2=tf.keras.layers.Dense(vocab_size)

    def call(self,x,features,hidden):

        context_vector,attention_weights=self.attention(features,hidden)

        x=self.embedding(x)

        x=tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)

        output,state,_=self.lstm(x)

        x=self.fc1(output)

        x=tf.reshape(x,(-1,x.shape[2]))

        x=self.fc2(x)

        return x,state,attention_weights

# Load attention tokenizer

with open(ATTN_TOKENIZER_PATH,"rb") as f:
    attn_tokenizer=pickle.load(f)

with open(ATTN_METADATA_PATH,"rb") as f:
    meta=pickle.load(f)

attn_max_length=meta["max_length"]
vocab_size=meta["vocab_size"]

attn_index_word={v:k for k,v in attn_tokenizer.word_index.items()}

encoder_model=Encoder(EMBED_DIM)
decoder_model=Decoder(EMBED_DIM,UNITS,vocab_size)

dummy_img=tf.zeros((1,224,224,3))
dummy_feat=encoder_model(dummy_img)

dummy_hidden=tf.zeros((1,UNITS))
dummy_input=tf.zeros((1,1),dtype=tf.int32)

decoder_model(dummy_input,dummy_feat,dummy_hidden)

encoder_model.load_weights(ENCODER_WEIGHTS)
decoder_model.load_weights(DECODER_WEIGHTS)

def generate_attention_caption(img_path):

    img=load_img(img_path,target_size=(224,224))
    img=img_to_array(img)
    img=preprocess_input(img)
    img=tf.expand_dims(img,0)

    features=encoder_model(img)

    start=attn_tokenizer.word_index["startseq"]
    end=attn_tokenizer.word_index["endseq"]

    seq=[start]

    hidden=tf.zeros((1,UNITS))

    for _ in range(attn_max_length):

        dec_input=tf.expand_dims([seq[-1]],1)

        preds,hidden,_=decoder_model(dec_input,features,hidden)

        word=np.argmax(preds[0])

        seq.append(word)

        if word==end:
            break

    caption=[attn_index_word.get(i) for i in seq if i not in [start,end]]

    return " ".join(caption)

# ---------------------------
# BLIP MODEL
# ---------------------------

from transformers import BlipProcessor,BlipForConditionalGeneration

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor=BlipProcessor.from_pretrained(BLIP_MODEL_NAME)

blip_model=BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL_NAME
).to(device)

blip_model.eval()

def generate_blip_caption(img_path):

    image=Image.open(img_path).convert("RGB")

    inputs=processor(images=image,return_tensors="pt").to(device)

    with torch.no_grad():

        output=blip_model.generate(**inputs,max_length=20,num_beams=3)

    caption=processor.decode(output[0],skip_special_tokens=True)

    return caption