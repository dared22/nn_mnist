from lowrank.layers.dense_layer import DenseLayer

def test_initialization():
    layer  = DenseLayer(10, 5, rank=3)  # Adjust rank as needed

    # Check if weights are initialized with He initialization
    assert layer.weight.shape == (10, 5)
    assert layer.bias.shape == (5,)

# class TestDenseLayer(unittest.TestCase):

#     def setUp(self):
#         self.input_size = 10
#         self.output_size = 5
#         self.layer = DenseLayer(self.input_size, self.output_size, rank=3)  # Adjust rank as needed

#     def test_initialization(self):
#         # Check if weights are initialized with He initialization
#         self.assertEqual(self.layer.weight.shape, (self.input_size, self.output_size))
        
#         # Check if biases are initialized from a standard normal distribution
#         self.assertEqual(self.layer.bias.shape, (self.output_size,))

#     def test_forward_pass(self):
#         # Create a dummy input
#         x = torch.randn(1, self.input_size)
#         output = self.layer(x)
        
#         # Check if output is computed
#         self.assertIsNotNone(output)

#     def test_output_shape(self):
#         # Create a dummy input
#         x = torch.randn(1, self.input_size)
#         output = self.layer(x)
        
#         # Check the shape of output
#         self.assertEqual(output.shape, (1, self.output_size))

#     def test_different_input_sizes(self):
#         # Test with a different input size
#         x = torch.randn(1, self.input_size * 2)
#         self.layer = DenseLayer(self.input_size * 2, self.output_size, rank=3)  # Adjust rank as needed
#         output = self.layer(x)
#         self.assertEqual(output.shape, (1, self.output_size))

#     def test_gradient_flow(self):
#         x = torch.randn(1, self.input_size, requires_grad=True)
#         output = self.layer(x)
#         output.sum().backward()
#         self.assertIsNotNone(x.grad)

# if __name__ == '__main__':
#     unittest.main()
