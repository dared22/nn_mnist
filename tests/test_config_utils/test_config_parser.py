from lowrank.config_utils.config_parser import ConfigParser

def test_parser():
	config_parser = ConfigParser("tests/data/config_ex_ffn.toml")

	config_parser.load_config()

	assert config_parser.config is not None
