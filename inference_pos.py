import nltk
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import shutil
from model import Model
from data_pos import POSTags
import numpy as np

pos_obj = POSTags()

model = Model(model_name = 'zigzag_resnet')

# Run the below cell after loading the best model 
#model.load_state_dict(torch.load('./models/best_model.pth'))

def get_sentiment(user_input):
    if not os.path.isdir('.user'):
        os.mkdir('.user')
    user_input = ''.join(user_input).replace('\n', '').split('.')
    try:
        user_input.remove('')
    except:
        pass
    itr = 1
    if len(user_input) >= 3:
        
        max_len = 0
        min_len = 999999999
        arrs = []
        arr_lens = []
        for user_inp in user_input:
            arr, arr_len = pos_obj.get_tags(user_inp) 
            arrs.append(arr)
            arr_lens.append(arr_len)
            if arr_len > max_len:
                max_len = arr_len
            if arr_len < min_len:
                min_len = arr_len
        
        arrs_ = []
        for i in range(len(arrs)):
            arr = arrs[i]
            arr.extend([0]*(max_len - arr_lens[i]))
            arrs_.append(arr)
            
        
        for i in range(len(user_input) - 2):
            arr1, arr1_len = pos_obj.get_tags(user_input[i]) 
            arr2, arr2_len = pos_obj.get_tags(user_input[i + 1]) 
            arr3, arr3_len = pos_obj.get_tags(user_input[i + 2]) 

            max_len = max([arr1_len, arr2_len, arr3_len])
            min_len = min([arr1_len, arr2_len, arr3_len])

            arr1.extend([0]*(max_len - arr1_len))
            arr2.extend([0]*(max_len - arr2_len))
            arr3.extend([0]*(max_len - arr3_len))

            # Stack the arrays vertically
            data = np.vstack([arr1, arr2, arr3])

            # Create a contour plot
            plt.contourf(data, cmap='hsv')

            # Set Plot Axis as OFF
            plt.axis('off')

            plt.savefig(f"./.user/user_{itr}.png",bbox_inches='tight', pad_inches=0, dpi=100)
            itr += 1

            plt.close()

    # Set the root directory of your images
    root_dir = "./.user/"

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = os.listdir(root_dir)

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            image_file = self.image_files[index]
            image_path = os.path.join(self.root_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image.to(device)

    # Define the transform to apply on the images
    
    mean = [0.0028, 0.0024, 0.0006]
    std = [0.0014, 0.0010, 0.0011]
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])

    # Create an instance of the custom dataset
    dataset = CustomImageDataset(root_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    predicted_classes = []
    count = 0
    ai_perc = 0
    human_perc = 0
    with torch.no_grad():
        for i, image in enumerate(dataloader, 0):
            count += 1
            image = image.to(device)
            output = model(image)
            probabilities = torch.sigmoid(output)
            ai_probability = probabilities.detach().cpu().numpy()[0][0]
            human_probability = probabilities.detach().cpu().numpy()[0][1]
            total_probability = ai_probability + human_probability
            ai_percentage = ai_probability / total_probability
            human_percentage = human_probability / total_probability
            ai_perc += ai_percentage
            human_perc += human_percentage
            class_labels = {0: 'AI', 1: 'Human'}
            predicted_class = torch.argmax(probabilities).item()
            predicted_classes.append(class_labels[predicted_class])
    shutil.rmtree(root_dir)
    return(f"AI: {round((ai_perc/count)*100, 3)}%; Human: {round((human_perc/count)*100, 3)}%")

# Get user input
while True:
    text = input("Enter a text to classify its sentiment (type 'quit' to exit): ")
    if text.lower() == 'quit':
        break
    sentiment = get_sentiment(text)
    print(f"Sentiment: {sentiment}")