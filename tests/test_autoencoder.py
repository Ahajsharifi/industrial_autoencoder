import torch
from models.autoencoder import AutoEncoder

def main():
    model = AutoEncoder()
    model.eval()
    x = torch.randn (1,3,128,128)
    with torch.no_grad():

        y= model(x)

    print(x.shape)
    print(y.shape)

    assert y.shape == (1,3,128,128)
if __name__ == "__main__":
    main()