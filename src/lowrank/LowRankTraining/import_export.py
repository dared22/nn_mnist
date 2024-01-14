import torch



class DataMover:
    """
    A class for moving neural network models to and from a specified file path.

    This class provides functionality to import and export the state dictionaries
    of PyTorch neural network models. It's designed to work with models defined
    using the PyTorch library.

    Attributes:
        path (str): The file path where the model's state dictionary is stored or will be stored.

    Methods:
        imprt: Import a model's state dictionary from the specified path.
        export: Export a model's state dictionary to the specified path.
    """

    def __init__(self, path):
        """
        Initializes the DataMover with a specified file path.

        Args:
            path (str): The file path where the model's state dictionary is stored or will be stored.
        """
        self.path = path

    def imprt(self, nn):
        """
        Imports a neural network model's state dictionary from the specified path.

        This method initializes a new model of the specified class and loads its state
        dictionary from the file path set in the DataMover instance.

        Args:
            nn (class): The neural network class to be instantiated and loaded.

        Returns:
            The loaded neural network model with its state dictionary imported from the file.
        """
        model = nn()
        model.load_state_dict(torch.load(self.path))
        return model
    
    def export(self, trained_nn):
        """
        Exports a trained neural network model's state dictionary to the specified path.

        This method saves the state dictionary of the provided neural network model
        to the file path set in the DataMover instance.

        Args:
            trained_nn (torch.nn.Module): The trained neural network model whose state dictionary is to be saved.
        """
        torch.save(trained_nn.state_dict(), self.path)
