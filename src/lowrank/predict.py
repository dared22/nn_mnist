import torch
import torchvision.transforms as T
from PIL import Image
from lowrank.training.MNIST_downloader import Downloader

def predict(model, pic_nr, test):
    mnist_tensors = test.data
    mnist_labels = test.targets
    tensor = mnist_tensors[pic_nr]
    label = mnist_labels[pic_nr] 
    with torch.no_grad():  # Disable gradient calculations for prediction
        # Preprocess the tensor (e.g., normalization, adding batch dimension)
        tensor = tensor.float() / 255.0  # Assuming MNIST images are in grayscale
        tensor = tensor.unsqueeze(0)  # Adding a batch dimension

            # Make the prediction
        output = model(tensor)
        prediction = output.argmax()  # Getting the most likely class
    return prediction.item(), label.item()

def show_image(tensor):
    transform = T.ToPILImage()

    # convert the tensor to PIL image using above transform
    img = transform(tensor)

    # display the PIL image
    img.show()

    