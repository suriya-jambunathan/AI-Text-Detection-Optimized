# A Robust and Accurate Algorithm for Detecting AI-Generated Text

![Python](https://img.shields.io/badge/Python-3.10-blue)

## Team Members: 
### Suriya Prakash Jambunathan (sj3828), NYU
### Ashwath Shankarnarayan (as16494), NYU


## Why Image Approach over Text

1. NLP models have huge number of parameters (eg BERT- 110 million)
2. Image classification models have reasonably less parameters
3. A lot of image classification models have better and consistent accuracy scores
4. Less Training time for image classification models
5. Observe the spatial relation between the sentences.


## Model details

We pursued a novel approach to address the challenge of optimizing the ResNet-18 architecture . Our strategy  incorporated a diversified channel size progression. This approach involved starting with 64 channels, gradually increasing to 1024, then gradually decreasing it to 64, then again increasing it to 1024. This configuration introduced more variety into the model, utilizing several distinct block structures, in contrast to the four unique block structures in the original ResNet-18. Our results indicated that the ZigZag model outperformed the other architectures in terms of accuracy.
Alternatively, we also trained our image-based data on GoogleNet, which is a well-known model for image classification problems.


## Text embedding in Image

### Parts of Speech
1. Split the input paragraph into overlapping 3-sentences granularity.
2. Assign an id to each of the word in the sentences using Parts of Speech (POS) tags.
3. Pad zeros to POS Tags to make the embedding equal for all sentences.
4. Vertically stack the PAD Tags of the three sentences.
5. Plot a contour map of the vertical stack (final embedding).

![alt text](/results/text_embedding_sample.png)

### Universal Sentence Encoder
1. Split the input paragraph into overlapping 3-sentences granularity.
2. Apply the Universal Sentence Encoder (USE) on the 3-sentences to get a 512-long Embedding.
3. Scale the embeddings to the range of 0-255 to emulate image pixels.
4. The First Fully Connected Layer in our modified ResNet-18 model will take care of the image generation by scaling 512 to 768 and later reshaping it to 3 16x16 images.

![alt text](/results/text_embedding_use_sample.png)

## Training and Testing on HC3 Data

https://huggingface.co/datasets/Hello-SimpleAI/HC3/blob/main/README.md

```
from model import Model
from data_pos import Data as POSData
from data_use import Data as USEData
from torch.utils.data import DataLoader

use_data = 'use'

if use_data == 'pos':
    # Initialize Data object with the CSV file name
    data_obj = POSData(csv_name='./data/HC3.csv')

    # Save POS tagged images for the 'ai' category
    data_obj.save_pos_tagged_images('ai', images_dir='./data/numsent_3/')

    # Save POS tagged images for the 'human' category
    data_obj.save_pos_tagged_images('human', images_dir='./data/numsent_3/')

    # Save torch data batches to the specified folder path
    data_obj.save_torch_data_batches(folder_path='./data/numsent_3/batches/')

    # Get train, test, and validation datasets
    train_set, test_set, val_set = data_obj.get_train_test_val_data()

    # Initialize the model
    model = Model('zigzag_resnet')

elif use_data == 'use':
    # Initialize Data object with the CSV file name
    data_obj = USEData(csv_name='./data/HC3.csv')

    # Get train, test, and validation datasets
    train_set, test_set, val_set = data_obj.get_train_test_val_data()

    # Initialize the model
    model = Model('text_resnet')

# Set the batch size
batch_size = 32

# Create a DataLoader for the training set to handle batch loading and shuffling
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False)

# Create a DataLoader for the validation set
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=False)

# Create a DataLoader for the test set
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=False)

# Train the model for the specified number of epochs using the training and validation loaders
model.train(num_epochs=100, train_loader=train_loader, val_loader=valid_loader)

# Test the trained model using the test loader
model.test(dataloader=test_loader)

```
## Results 

### 1. Text approach (BERT) 

![alt text](/results/bert_model_inference.jpg)

### 2. Image approach 
### POS Tagging (GoogleNet/ResNet)
![alt text](/results/image_model_performance.png)

### USE Embedding (ResNet)
![alt text](/results/use_model_inference.jpg)

### Observation

The dataset HC3 works very well on NLP models. We trained the HC3 dataset on a BERT pre-trained tokenizer and model and we were able to achieve a test accuracy of 99% and along with that we tested the model and compared it with other AI classifiers such as GPTZero and OpenAI classifier and the BERT model performed  better than both of the other classifiers.
The image approach where we are making use of a CNN model, we were able to achieve a test accuracy of 93-94% (within the dataset), although the model did not generalise very well for outside of the dataset. This is due to the fact that we have less amount of variety data points which is a bottleneck.

The Text-Based BERT Model generalized very well and performed near-perfectly on multiple LLM generated data.
The Image-Based Model learnt the pattern in the data very well, and performed nearly 93% accurate on the HC3 Dataset. Due to lack of variety of LLM-Generated data and their corresponding image-embeddings, the model did not generalize well. With better quality data in large numbers, the model should generalize and predict well.  


## Inference on custom text

### Parts of Speech
```
python3 inference_pos.py
```

### Universal Sentence Encoder
```
python3 inference_use.py
```
