# A Robust and Accurate Algorithm for Detecting AI-Generated Text

![Python](https://img.shields.io/badge/Python-3.10-blue)

## Training and Testing on HC3 Data

https://huggingface.co/datasets/Hello-SimpleAI/HC3/blob/main/README.md

Please download the hc3.csv and place in the data directory before training.

Using Parts of Speech (POS) Method
```
python3 main.py --method POS --train --num_epochs 100
```

Using Universal Sentence (USE) Method
```
python3 main.py --method USE --train --num_epochs 100
```

## Model Inference

Using Parts of Speech (POS) Method
```
python3 main.py --method POS --model_path './model_store/best_pos_model.pth' --infer
```

Using Universal Sentence (USE) Method
```
python3 main.py --method USE --model_path './model_store/best_use_model.pth' --infer
```

## Why Image Approach over Text

1. NLP models have huge number of parameters (eg BERT- 110 million)
2. Image classification models have reasonably less parameters
3. A lot of image classification models have better and consistent accuracy scores
4. Less Training time for image classification models
5. Observe the spatial relation between the sentences.

## Model details

The novel approach to optimizing the ResNet-18 architecture involved incorporating a diversified channel size progression. This approach starts with a fully connected layer of size 768. The output of this layer is reshaped to 3 16x16 images stacked on top of each other. This image is passed to a convolutional layer with 64 channels, gradually increased to 256, then decreased back to 64, and finally increased again to 256. By introducing this configuration, the model introduced more variety and utilized several distinct block structures, in contrast to the four unique block structures in the original ResNet-18. The results indicated
that the ZigZag model outperformed other architectures in terms of accuracy.

![alt text](/results/ZigZagTextNet.png)

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

## Results 

### 1. Text approach (BERT) 

#### Accuracy Plot
![alt text](/results/BERT_accuracy_plt.png)

#### Inference
![alt text](/results/bert_model_inference.jpg)

### 2. Image approach (Universal Sequence Encoder - ZigZag ResNet)

#### Accuracy Plot
![alt text](/results/use_embedding_resnet_accuracy_plot.jpg)

#### Inference
![alt text](/results/use_model_inference.jpg)

### Conclusion

First, we trained a BERT-based model on the HC3 dataset, and achieved an impressive test accuracy of 99%. This indicates that the BERT model effectively learned the patterns and features in the dataset, leading to highly accurate predictions. Next, we explored an image-based approach using a CNN model. The CNN model was trained on the HC3 dataset, and achieved a test accuracy of 93-94% within the dataset. However, we observed that the CNN model did not generalize well to data points outside of the dataset. This limitation can be attributed to the lack of variety in the data points used for training the model.Finally, we investigated
the performance of the ResNet model with USE encoder. The ResNet model demonstrated excellent performance and generalization, achieving a test accuracy of 97%. In conclusion, our novel approach of using our modified ResNet with the Universal Sequence Encoder has performed very well and generalized very well on outside points as well. This approach is better than the traditional BERT model since the inference time and train times are much faster than the BERT model. It takes around 40 minutes to train one epoch on the BERT model. At the same time our modified ResNet trains within 1 minute for one epoch and has a much faster inference time. We also curated 100 data points from different generative AI bots to test it on real world data. Our model performed very well on those points too with an accuracy of 85%.
