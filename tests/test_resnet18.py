"""Quick architecture checks, no CIFAR-10 download or training needed."""
import unittest
import torch
from train_resnet18 import build_model


class ResNet18Tests(unittest.TestCase):
    def test_cifar_input_output_shape(self):
        model = build_model().eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 10))

    def test_cifar_stem(self):
        model = build_model()
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)


if __name__ == "__main__":
    unittest.main()
