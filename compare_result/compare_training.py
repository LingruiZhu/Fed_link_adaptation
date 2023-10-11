import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_history(history_file:str):
    history_loss = dict()
    with h5py.File(history_file, 'r') as hf:
        for key in hf.keys():
            history_loss[key] = hf[key][:] 
    return history_loss


class TraingHistoryPlotter:
    def __init__(self, file_path:str, label:str, line_format:str) -> None:
        self.file_path = file_path
        self.label = label
        self.line_format = line_format
        self.historys = read_history(file_path)


def plot_training_loss(plotter_list:list):
    number_epochs = 500
    
    plt.figure()
    for plotter in plotter_list:
        plt.semilogy(plotter.historys["total_loss"], plotter.line_format, label=plotter.label, markevery=20)
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in plotter_list:
        plt.semilogy(plotter.historys["reconstruction_loss"], plotter.line_format, label=plotter.label, markevery=20)
    # plt.semilogy(ae_history["loss"][:number_epochs], "r-", label="AE")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/reconstruction_loss.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in plotter_list:
        plt.semilogy(plotter.historys["codebook_loss"], plotter.line_format, label=plotter.label, markevery=20)
    plt.xlabel("Epochs")
    plt.ylabel("Codebook Loss")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/codebook.pdf", format="pdf", bbox_inches="tight")
    
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in plotter_list:
        plt.plot(plotter.historys["num_active_embeddings"], plotter.line_format, label=plotter.label, markevery=20)
    plt.xlabel("Epochs")
    plt.ylabel("Number of Updated Embeddings")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/num_active_embeddings.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in plotter_list:
        plt.semilogy(plotter.historys["learning_rates"], plotter.line_format, label=plotter.label, markevery=20)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rates")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/learning_rates.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    
if __name__ == "__main__":
    vq_vae_history_path = "training_history/vq_vae/vq_vae_input_40_latent_20_num_embeddings_128_init_random_with_BN_False_RMSprop.h5"
    vq_vae_history_plotter = TraingHistoryPlotter(file_path=vq_vae_history_path, label="VQ-VAE", line_format="b-x")
    
    vq_vae_kmpp_history_path = "training_history/vq_vae_kmpp_init/vq_vae_input_40_latent_20_num_embeddings_128_init_kmpp_with_BN_False_RMSprop.h5"
    vq_vae_kmpp_history_plotter = TraingHistoryPlotter(file_path=vq_vae_kmpp_history_path, label="VQ-VAE-KMPP", line_format="m-+")
    
    vq_vae_ema_history_path = "training_history/vq_vae_ema/vq_vae_ema_input_40_latent_20_num_embeddings_128_init_random_ema_decay_0.99_beta_0.25.h5"
    vq_vae_ema_history_plotter = TraingHistoryPlotter(file_path=vq_vae_ema_history_path, label="VQ-VAE-EMA", line_format="g-o")
    
    vq_vae_ema_kmpp_history_path = "training_history/vq_vae_ema_kmpp_init/vq_vae_ema_input_40_latent_20_num_embeddings_128_init_kmpp_ema_decay_0.99_beta_0.25.h5"
    vq_vae_ema_kmpp_history_plotter = TraingHistoryPlotter(file_path=vq_vae_ema_kmpp_history_path, label="VQ-VAE-KMPP", line_format="r-s")
    
    plotter_list = [vq_vae_history_plotter,
                    vq_vae_kmpp_history_plotter,
                    vq_vae_ema_history_plotter,
                    vq_vae_ema_kmpp_history_plotter]
    
    print(vq_vae_ema_history_plotter.historys["num_active_embeddings"])
    
    plot_training_loss(plotter_list)
    