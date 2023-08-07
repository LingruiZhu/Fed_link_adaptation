import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_history(history_file:str):
    history_loss = dict()
    with h5py.File(history_file, 'r') as hf:
        for key in hf.keys():
            history_loss[key] = hf[key][:] 
    return history_loss


def plot_training_loss(vq_vae_history_path, vq_vae_ema_histroy_path, ae_history_path):
    vq_vae_history = read_history(vq_vae_history_path)
    vq_vae_ema_history = read_history(vq_vae_ema_histroy_path)
    ae_history = read_history(ae_history_path)
    
    print(ae_history.keys())
    
    number_epochs = 400
    
    plt.figure()
    plt.semilogy(vq_vae_history["loss"][:number_epochs], "b-", label="VQ-VAE")
    plt.semilogy(vq_vae_ema_history["loss"][:number_epochs], "g-", label="VQ-VAE-EMA")
    # plt.semilogy(vq_vae_history["reconstruction_loss"][:number_epochs], "b--", label="VQ-VAE reconsctruction loss")
    # plt.semilogy(vq_vae_ema_history["reconstruction_loss"][:number_epochs], "g--", label="VQ-VAE-EMA reconstruction loss")
    # plt.semilogy(vq_vae_history["codebook_loss"][:number_epochs], "b .", label="VQ-VAE embedding loss")
    # plt.semilogy(vq_vae_ema_history["vqvae_loss"][:number_epochs], "g .", label="VQ-VAE-EMA embedding loss")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    plt.semilogy(vq_vae_history["reconstruction_loss"][:number_epochs], "b-", label="VQ-VAE")
    plt.semilogy(vq_vae_ema_history["reconstruction_loss"][:number_epochs], "g-", label="VQ-VAE-EMA")
    # plt.semilogy(ae_history["loss"][:number_epochs], "r-", label="AE")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/reconstruction_loss.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    plt.semilogy(vq_vae_history["codebook_loss"][:number_epochs], "b-", label="VQ-VAE")
    plt.semilogy(vq_vae_ema_history["vqvae_loss"][:number_epochs], "g-", label="VQ-VAE-EMA")
    plt.xlabel("Epochs")
    plt.ylabel("Codebook Loss")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/embedding_loss.pdf", format="pdf", bbox_inches="tight")
    
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    plt.semilogy(vq_vae_history["num_active_embeddings"][:number_epochs], "b-", label="VQ-VAE")
    plt.semilogy(vq_vae_ema_history["num_active_embeddings"][:number_epochs], "g-", label="VQ-VAE-EMA")
    plt.xlabel("Epochs")
    plt.ylabel("Number of Updated Embeddings")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/num_active_embeddings.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    plt.semilogy(vq_vae_history["learning_rates"][:number_epochs], "b-", label="VQ-VAE")
    plt.semilogy(np.mean(vq_vae_ema_history["learning_rates"][:number_epochs], axis=-1), "g-", label="VQ-VAE-EMA")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rates")
    plt.legend()
    plt.grid()
    plt.savefig("figures/training/learning_rates.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    
if __name__ == "__main__":
    vq_vae_loss = "training_history/vq_vae/vq_vae_input_40_latent_20_num_embeddings_256_with_BN_False_RMSprop.h5"
    vq_vae_ema_loss = "training_history/vq_vae_ema/vq_vae_ema_input_40_latent_20_num_embeddings_256_ema_decay_0.99_beta_0.25.h5"
    ae_loss = "training_history/ae/vq_vae_ema_input_40_latent_10_optimizer_RMSprop.h5"
    plot_training_loss(vq_vae_loss, vq_vae_ema_loss, ae_loss)
    