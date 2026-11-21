import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import CNN, CLASSES, evaluate, parse_args, split_indices


class ModelTests(unittest.TestCase):
    def test_forward_shape(self):
        self.assertEqual(tuple(CNN()(torch.zeros(2, 3, 32, 32)).shape), (2, 10))

    def test_split_reproducible_disjoint(self):
        a, b = split_indices(100, 0.2, 42)
        c, d = split_indices(100, 0.2, 42)
        self.assertEqual((a, b), (c, d))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((len(a), len(b)), (80, 20))

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            split_indices(10, 1.0, 42)

    def test_evaluation_counts(self):
        data = TensorDataset(torch.zeros(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))
        metrics, preds, labels = evaluate(CNN(), DataLoader(data, batch_size=2),
                                           torch.device("cpu"), nn.CrossEntropyLoss(), True)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(len(preds), 4)
        self.assertEqual(len(labels), 4)
        self.assertTrue(0 <= metrics["accuracy"] <= 1)

    def test_class_count(self):
        self.assertEqual(len(CLASSES), 10)

    def test_defaults(self):
        args = parse_args([])
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.workers, 0)


if __name__ == "__main__":
    unittest.main()
