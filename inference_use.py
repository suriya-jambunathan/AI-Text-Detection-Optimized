import torch
from torch.utils.data import Dataset, DataLoader
from model import Model
from data_use import Data

model = Model(model_name = 'text_resnet')

data_obj = Data()

# Run the below cell after loading the best model 
#model.load_state_dict(torch.load('./models/best_model.pth'))

class USE_Inference_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        # Convert the sublist to a PyTorch tensor
        tensor = torch.tensor(self.data[index])
        return tensor
    
    def __len__(self):
        return len(self.data)
    
def get_sentiment(user_input):
    user_input = ''.join(user_input).replace('\n', '').split('.')

    user_input_embeddings = data_obj.process_paragraph(user_input, num_sentences = 3)

    scaled_user_embeddings = [data_obj.map_values_to_range(user_input_embedding, max_val = 255) for user_input_embedding in user_input_embeddings]

    dataset = USE_Inference_Dataset(scaled_user_embeddings)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.eval()
    predicted_classes = []
    count = 0
    ai_perc = 0
    human_perc = 0

    # Check if CUDA is available and use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return(f"AI: {round((ai_perc/count)*100, 3)}%; Human: {round((human_perc/count)*100, 3)}%")

# Get user input
while True:
    text = input("Enter a text to classify its sentiment (type 'quit' to exit): ")
    if text.lower() == 'quit':
        break
    sentiment = get_sentiment(text)
    print(f"Sentiment: {sentiment}")