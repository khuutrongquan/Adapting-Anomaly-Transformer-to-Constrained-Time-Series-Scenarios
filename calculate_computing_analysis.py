import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model.AnomalyTransformer import AnomalyTransformer

def count_flops(model, input_size):
    model.eval()
    dummy_input = torch.randn(*input_size).cuda()  # input on GPU
    flops = FlopCountAnalysis(model, (dummy_input,))  # wrap in tuple
    return flops.total()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Define model
    model = AnomalyTransformer(
        win_size=100,
        enc_in=2,
        c_out=2,
        e_layers=3
    ).cuda()

    #Flops count
    flops = count_flops(model, (1, 100, 2))
    print(f"Total FLOPs: {flops / 1e9:.4f} GFLOPs")

    #Parameter count
    num_params = count_params(model)
    print(f"Trainable Parameters: {num_params / 1e6:.2f}")


if __name__ == "__main__":
    main()
