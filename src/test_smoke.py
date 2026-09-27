import torch
from models import build_model


def test_forward_shapes():
    x = torch.randn(4, 1, 28, 28)
    for name in ["mlp", "cnn", "transfer"]:
        for nc in (10, 47):
            model = build_model(name, num_classes=nc)
            out = model(x)
            assert out.shape == (4, nc), f"{name}/{nc}: got {out.shape}"
            assert not torch.isnan(out).any(), f"{name}: NaN output"


def test_transfer_partial_unfreeze():
    model = build_model("transfer")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert "fc.weight" in trainable and "conv1.weight" in trainable
    assert "layer4.1.conv2.weight" in trainable  # layer4 unfrozen
    assert "layer1.0.conv1.weight" not in trainable  # layer1-3 stay frozen


if __name__ == "__main__":
    test_forward_shapes()
    test_transfer_partial_unfreeze()
    print("smoke tests passed")
