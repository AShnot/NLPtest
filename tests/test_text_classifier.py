import unittest
import torch
from math_app.text_classifier import SimpleNN

class TestSimpleNN(unittest.TestCase):
    def test_forward_shape(self):
        model = SimpleNN(input_dim=10, num_classes=3)
        x = torch.randn(5, 10)
        out = model(x)
        self.assertEqual(out.shape, (5, 3))

    def test_training_step(self):
        model = SimpleNN(input_dim=4, num_classes=2)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        self.assertIsInstance(loss.item(), float)

if __name__ == '__main__':
    unittest.main() 