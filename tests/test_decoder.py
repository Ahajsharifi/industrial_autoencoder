from models.decoder import Decoder
import torch

def main () :

    model = Decoder()
    model.eval()

    bottleneck = torch.randn (1,256,8,8)

    skips = [
        torch.randn(1,32,128,128),
        torch.randn(1,64,64,64),
        torch.randn(1,128,32,32),
        torch.randn(1,256,16,16),
    ]


    with torch.no_grad():

        reconstruction = model(bottleneck,skips)

    print(reconstruction.shape)

if __name__ == "__main__" :
    main()