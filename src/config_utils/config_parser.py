import toml

class ConfigParser:
    def __init__(self, path):
        """
        Initialize the parser with the path to the TOML configuration file.

        Parameters
        ----------
        path : str
            The file path to the TOML configuration file.

        Attributes
        ----------
        path : str
            Stores the path to the configuration file.
        config : dict
            Stores the loaded configuration data.

        """
        self.path = path
        self.config = self.load_config()

    def load_config(self):
        """
        Load the configuration from the TOML file.

        Returns
        -------
        dict or None
            The configuration dictionary if file is successfully loaded, 
            otherwise None.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at the specified path.

        """
        try:
            with open(self.path, 'r') as file:
                return toml.load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

    def save(self):
        """
        Save the current configuration back to the TOML file.

        Raises
        ------
        Exception
            If an error occurs during file writing.

        """
        try:
            with open(self.path, 'w') as file:
                toml.dump(self.config, file)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def __getitem__(self, key):
        """
        Retrieve a value from the configuration.

        Parameters
        ----------
        key : str
            The key for the value to be retrieved from the configuration.

        Returns
        -------
        value
            The value associated with the specified key. Returns None if key is not found.

        """
        return self.config.get(key, None)

    def __setitem__(self, key, value):
        """
        Set a value in the configuration.

        Parameters
        ----------
        key : str
            The key for which the value needs to be set in the configuration.
        value
            The value to be set for the specified key.

        """
        self.config[key] = value

    def get_config(self):
        """
        Return the entire configuration.

        Returns
        -------
        dict
            The complete configuration dictionary.

        """
        return self.config