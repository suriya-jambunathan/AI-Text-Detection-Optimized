import os
import ast
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
from multiprocessing import Pool
from tqdm import tqdm
import tensorflow_hub as hub

import matplotlib
matplotlib.use('Agg')

class Data():
    """
    A container for the data used to train and evaluate the POS tagging model.
    """

    def __init__(self, csv_name =  None):
        """
        Initializes the class and reads the data from the CSV file.
        """
        if csv_name != None:
            self.csv_name = csv_name
            self._read_csv()
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.use_embed_model = hub.load(module_url)

    def _read_csv(self):
        """
        Read the CSV file and store the data in the following variables:

        * `self.df`: The Pandas DataFrame object.
        * `self.human_paragraphs`: A list of lists of strings, where each inner list represents a paragraph of human answers.
        * `self.chatgpt_paragraphs`: A list of lists of strings, where each inner list represents a paragraph of ChatGPT answers.
        """

        self.df = pd.read_csv(self.csv_name)

        # Create a list of lists of strings, where each inner list represents a paragraph of human/chatgpt answers.
        self.human_paragraphs = [''.join(ast.literal_eval(human_paragraph)).replace('\n', '').split('.') 
                                for human_paragraph in list(self.df['human_answers'])]
        self.chatgpt_paragraphs = [''.join(ast.literal_eval(chatgpt_paragraph)).replace('\n', '').split('.') 
                                for chatgpt_paragraph in list(self.df['chatgpt_answers'])]

    def process_paragraph(self, paragraph, num_sentences):
        embeddings = []
        for i in range(0, len(paragraph)-num_sentences):
            mini_paragraph = '\n'.join(paragraph[i:i+num_sentences])

            # Use the Universal Sentence Encoder to get the embedding
            use_embedding = self.use_embed_model([mini_paragraph])

            # Process the embedding values and save the image
            embeddings.append(use_embedding[0].numpy().tolist())
        return(embeddings)
    
    def map_values_to_range(self, arr, max_val = 255):
        minimum = np.min(arr)
        maximum = np.max(arr)
        value_range = maximum - minimum
        mapped_values = []
        for value in arr:
            value -= minimum
            value /= value_range
            value *= max_val
            mapped_values.append(value)
        return mapped_values
        
    def get_embeddings(self, name, num_sentences, limit = 150000):
        if name.lower() == 'human':
            paragraphs = self.human_paragraphs
        else:
            paragraphs = self.chatgpt_paragraphs
        embeddings = []
        for paragraph in tqdm(paragraphs):
            embeddings.extend(self.process_paragraph(paragraph, num_sentences))
            if len(embeddings) >= limit:
                break
        embeddings = embeddings[:limit]
        return(embeddings)
    
    def get_train_test_val_data(self, limit_per_class = 150000):
        """
        Gets the training, testing, and validation datasets.

        Returns:
            A tuple of the training, testing, and validation datasets.
        """
        chatgpt_embeddings = self.get_embeddings(name = 'ai', num_sentences = 3)
        human_embeddings   = self.get_embeddings(name = 'human', num_sentences = 3)

        labels = []
        labels.extend(['ai']*limit_per_class)
        labels.extend(['human']*limit_per_class)

        embeddings = []
        embeddings.extend(chatgpt_embeddings)
        embeddings.extend(human_embeddings)

        embeddings_ = []
        for embedding in embeddings:
            embeddings_.append(self.map_values_to_range(embedding, max_val = 255))
        
        tembeddings = torch.from_numpy(np.array(embeddings_))

        labels_ = []
        for label in labels:
            if label == 'ai':
                labels_.append(0)
            else:
                labels_.append(1)

        transform = transforms.Compose([transforms.ToTensor()])

        dataset = USEDataset(tembeddings, labels_)

        train_set, test_set = torch.utils.data.random_split(dataset, [250000, 50000])
        train_set, val_set  = torch.utils.data.random_split(train_set, [200000, 50000])

        return(train_set, test_set, val_set)

class USEDataset(Dataset):
    def __init__(self, tensors, labels, transform=None):
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        tensor = self.tensors[index]
        label = self.labels[index]
        
        # Apply data transformation if available
        if self.transform is not None:
            tensor = self.transform(tensor)
            
        label = torch.tensor(label)

        return tensor, label