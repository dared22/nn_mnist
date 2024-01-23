import torch
from torchvision import datasets, transforms

class Downloader:
    """
    A class to handle the downloading and saving of the MNIST dataset.

    This class downloads the MNIST dataset for both training and testing. It applies
    a basic transformation to the images (converting them to tensor format) and saves
    the datasets locally.

    Attributes
    ----------
    mnist_train : torchvision.datasets.mnist.MNIST
        The MNIST training dataset.
    mnist_test : torchvision.datasets.mnist.MNIST
        The MNIST test dataset.

    Methods
    -------
    get_data()
        Returns the MNIST training and test datasets.
    """

    def __init__(self) -> None:
        """
        Initializes and downloads the MNIST dataset.

        The MNIST dataset is downloaded and stored in the './data' directory.
        The images are transformed into tensor format. The datasets are then saved
        as '.pt' files in the './data' directory.
        """

        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Download and load the MNIST training data
        self.mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=transform)

        # Download and load the MNIST test data
        self.mnist_test = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

        # Save the datasets
        torch.save(self.mnist_train, './data/mnist_train.pt')
        torch.save(self.mnist_test, './data/mnist_test.pt')

    def get_data(self):
        """
        Get the downloaded MNIST training and test datasets.

        Returns
        -------
        tuple
            A tuple containing the MNIST training dataset and test dataset.
        """
        
        train = self.mnist_train
        test = self.mnist_test
        return train, test



        
