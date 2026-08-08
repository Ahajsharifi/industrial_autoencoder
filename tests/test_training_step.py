# import torch.nn as nn
# import torch
# from models.autoencoder import AutoEncoder


# def main():


#     device = ("cuda"if torch.cuda.is_available() else "cpu")

#     model = AutoEncoder().to(device)

#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

#     criterion = nn.MSELoss()

#     x = torch.randn(4,3,128,128).to(device)


#     before = (model.encoder.encoder1.block[0].weight.clone())

#     optimizer.zero_grad()

#     reconstruction = model(x)

#     loss = criterion(reconstruction,x)

#     print("loss before backward:", loss.item())

#     loss.backward()

#     optimizer.step()

#     after = (model.encoder.encoder1.block[0].weight)

#     changed = not torch.equal(before,after)

#     print(
#         "Weights changed:",
#         changed
#     )


#     assert changed, (
#         "Model parameters did not update"
#     )


#     print(
#         "Training step test passed"
#     )
    




# if __name__ == "__main__" :
#     main()