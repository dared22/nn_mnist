import torch
import torchvision.transforms as T
from PIL import Image

def predict(model, data):
    predictions = []
    with torch.no_grad():  # Disable gradient calculations for prediction
        for tensor in data:
            # Preprocess the tensor (e.g., normalization, adding batch dimension)
            tensor = tensor.float() / 255.0  # Assuming MNIST images are in grayscale
            tensor = tensor.unsqueeze(0)  # Adding a batch dimension

            # Make the prediction
            output = model(tensor)
            prediction = output.argmax()  # Getting the most likely class
            predictions.append(prediction.item())  # Storing the predicted number
    return predictions

def show_image(tensor):
    transform = T.ToPILImage()

    # convert the tensor to PIL image using above transform
    img = transform(tensor)

    # display the PIL image
    img.show()

    