import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def save_model(epochs, model,imgsize, optimizer, loss,output_dir,name="model_rofie.pth"):
    """
    Function to save the trained model to disk.
    """
    import os 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'imgsize':imgsize
                }, f'{output_dir}/{name}')