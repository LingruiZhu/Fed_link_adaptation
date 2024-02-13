import h5py
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def read_history(history_file:str):
    history_loss = dict()
    with h5py.File(history_file, 'r') as hf:
        for key in hf.keys():
            history_loss[key] = hf[key][:] 
    return history_loss


def save_tikz(tikz_path:str, tikz_code):
    with open(tikz_path, 'w') as tikz_file:
        tikz_file.write(tikz_code)

class TraingHistoryPlotter:
    def __init__(self, file_path:str, label:str, line_format:str) -> None:
        self.file_path = file_path
        self.label = label
        self.line_format = line_format
        self.historys = read_history(file_path)


class TrainingHistoryPlotter_multiple:
    def __init__(self, file_path_list, label:str, color:str) -> None:
        self.file_path_list = file_path_list
        self.label = label
        self.color = color
        self.history_list = list()
        for file_path in file_path_list:
            history_item = read_history(file_path)
            self.history_list.append(history_item)
    
    
    def get_training_information(self, key_word:str):
        info_list = list()
        for history_item in self.history_list:
            info_list.append(history_item[key_word])
        info_array = np.array(info_list)
        info_mean = np.mean(info_array, axis=0)
        info_std = np.std(info_array, axis=0) 
        return info_mean, info_std
    

def plot_training_loss(plotter_list:list):
    number_epochs = 500
    
    # plt.figure()
    # for plotter in plotter_list:
    #     plt.semilogy(plotter.historys["total_loss"], plotter.line_format, label=plotter.label, markevery=20)
    # plt.xlabel("Epochs")
    # plt.ylabel("Total Loss")
    # plt.legend()
    # plt.grid()
    
    # plt.figure()
    # plt.rcParams['font.size'] = 12
    # for plotter in plotter_list:
    #     plt.semilogy(plotter.historys["reconstruction_loss"], plotter.line_format, label=plotter.label, markevery=20)
    # # plt.semilogy(ae_history["loss"][:number_epochs], "r-", label="AE")
    # plt.xlabel("Epochs")
    # plt.ylabel("Reconstruction Loss")
    # plt.legend()
    # plt.grid()
    # plt.savefig("figures/training/reconstruction_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plt.figure()
    # plt.rcParams['font.size'] = 12
    # for plotter in plotter_list:
    #     plt.semilogy(plotter.historys["codebook_loss"], plotter.line_format, label=plotter.label, markevery=20)
    # plt.xlabel("Epochs")
    # plt.ylabel("Codebook Loss")
    # plt.legend()
    # plt.grid()
    # plt.savefig("figures/training/codebook.pdf", format="pdf", bbox_inches="tight")
    
    
    # plt.figure()
    # plt.rcParams['font.size'] = 12
    # for plotter in plotter_list:
    #     plt.plot(plotter.historys["num_active_embeddings"], plotter.line_format, label=plotter.label, markevery=20)
    # plt.xlabel("Epochs")
    # plt.ylabel("Number of Updated Embeddings")
    # plt.legend()
    # plt.grid()
    # plt.savefig("figures/training/num_active_embeddings.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in plotter_list:
        if plotter.label == "VQ-VAE" or "VQ-VAE-KMPP":
            plt.semilogy(plotter.historys["learning_rates"], plotter.line_format, label=plotter.label, markevery=20)
        else:
            learning_rate_average = np.mean(plotter.historys["learning_rates"], axis=1)
            plt.semilogy(learning_rate_average, plotter.line_format, label=plotter.label, markevery=20)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rates")
    # plt.legend()
    plt.grid()
    plt.savefig("figures/training/learning_rates.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    

def plot_with_intervals(plotter:TrainingHistoryPlotter_multiple, key_word:str, log_scale:bool=True, num_embeddings:int=64, plot_interval:int=1):
    mean, std = plotter.get_training_information(key_word)
    lower_bound = list()
    upper_bound = list()
    for mean_element, std_element in zip(mean, std):
        lower_bound.append(mean_element-std_element)
        if key_word == "num_active_embeddings" and mean_element + std_element > num_embeddings:
            upper_bound.append(num_embeddings)
        else:
            upper_bound.append(mean_element+std_element)
    x_axis = np.arange(len(lower_bound))[::plot_interval]
    
    if log_scale:
        plt.semilogy(x_axis, mean[::plot_interval], color=plotter.color, label=plotter.label)
    else:
        plt.plot(x_axis, mean[::plot_interval], color=plotter.color, label=plotter.label)
    plt.fill_between(x_axis, lower_bound[::plot_interval], upper_bound[::plot_interval], color=plotter.color, alpha=0.5)


def plot_training_multiple_models(list_plotter_multiple:list):
    # total loss
    plt.figure()
    for plotter in list_plotter_multiple:
        plot_with_intervals(plotter, "total_loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("figures/training/total_loss_intervals.pdf", format="pdf", bbox_inches="tight")
    tikz_code_total_loss = tikzplotlib.get_tikz_code()
    save_tikz("figures/training/total_loss_intervals.tikz", tikz_code_total_loss)
    # tikzplotlib.save('figures/training/total_loss_intervals.tikz', scale=0.5)

    
    # reconstruction_lopss
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in list_plotter_multiple:
        plot_with_intervals(plotter, "reconstruction_loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("figures/training/reconstruction_loss_intervals.pdf", format="pdf", bbox_inches="tight")
    tikz_code_reconstruction_loss = tikzplotlib.get_tikz_code()
    save_tikz("figures/training/reconstruction_loss_intervals.tikz", tikz_code_reconstruction_loss)
    # tikzplotlib.save('figures/training/reconstruction_loss_intervals.tikz', scale=0.5)

    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in list_plotter_multiple:
        plot_with_intervals(plotter, "codebook_loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Codebook Loss")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("figures/training/codebook_with_intervals.pdf", format="pdf", bbox_inches="tight")
    tikz_code_codebook_loss = tikzplotlib.get_tikz_code()
    save_tikz("figures/training/codebook_with_intervals.tikz", tikz_code_codebook_loss)
    # tikzplotlib.save('figures/training/codebook_with_intervals.tikz', scale=0.5)
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in list_plotter_multiple:
        plot_with_intervals(plotter, "num_active_embeddings", log_scale=False)
    plt.xlabel("Epochs")
    plt.ylabel("Number of Updated Embeddings")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("figures/training/num_active_embeddings_with_intervals.pdf", format="pdf", bbox_inches="tight")
    tikz_code_num_embeddings = tikzplotlib.get_tikz_code()
    save_tikz("figures/training/num_active_embeddings_with_intervals.tikz", tikz_code_num_embeddings)
    # tikzplotlib.save('figures/training/num_active_embeddings_with_intervals.tikz', scale=0.5)
    
    plt.figure()
    plt.rcParams['font.size'] = 12
    for plotter in list_plotter_multiple:
        plot_with_intervals(plotter, "latent_entropy", log_scale=False, plot_interval=10)
    plt.xlabel("Epochs")
    plt.ylabel("Entropy of latent variables")
    plt.legend()
    plt.grid()
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("figures/training/latent_entropy.pdf", format="pdf", bbox_inches="tight")
    tikz_code_latent_entropy = tikzplotlib.get_tikz_code()
    save_tikz("figures/training/latent_entropy.tikz", tikz_code_latent_entropy)
    # tikzplotlib.save('figures/training/latent_entropy.tikz', scale=0.5)
    
    # plt.figure()
    # plt.rcParams['font.size'] = 12
    # for plotter in list_plotter_multiple:
    #     plot_with_intervals(plotter, "learning_rates", log_scale=False)
    # plt.xlabel("Epochs")
    # plt.ylabel("Learning Rates")
    # plt.legend()
    # plt.grid()
    # plt.savefig("figures/training/learning_rates_with_intervals.pdf", format="pdf", bbox_inches="tight")
    plt.show()
     

def single_plot():
    vq_vae_history_path = "training_history_new/VQ_VAE_history/latent_dim_20/num_embeddings_64/dense_vq_vae_input_40_latent_20_num_embeddings_64_init_random_RMSprop_index_0.h5"
    vq_vae_history_plotter = TraingHistoryPlotter(file_path=vq_vae_history_path, label="VQ-VAE", line_format="b-x")
    
    vq_vae_kmpp_history_path = "training_history_new/VQ_VAE_KMPP_history/latent_dim_20/num_embeddings_64/dense_vq_vae_input_40_latent_20_num_embeddings_64_init_kmpp_RMSprop_index_0.h5"
    vq_vae_kmpp_history_plotter = TraingHistoryPlotter(file_path=vq_vae_kmpp_history_path, label="VQ-VAE-KMPP", line_format="m-+")
    
    vq_vae_ema_history_path_eager = "training_history_new/VQ_VAE_EMA_history/latent_dim_20/num_embeddings_64/dense_vq_vae_ema_input_40_latent_20_num_embeddings_64_init_random_RMSprop_ema_decay_0.99_beta_0.25_index_0.h5"
    vq_vae_ema_history_plotter_eagar = TraingHistoryPlotter(file_path=vq_vae_ema_history_path_eager, label="VQ-VAE-EMA", line_format="g-o")
    
    vq_vae_ema_kmpp_history_path = "training_history_new/VQ_VAE_EMA_KMPP_history/latent_dim_20_1st_experiment/dense_vq_vae_ema_input_40_latent_20_num_embeddings_64_init_kmpp_RMSprop_ema_decay_0.99_beta_0.25_index_0.h5"
    vq_vae_ema_kmpp_history_plotter = TraingHistoryPlotter(file_path=vq_vae_ema_kmpp_history_path, label="VQ-VAE-EMA-KMPP", line_format="r-s")
    
    plotter_list = [vq_vae_history_plotter, vq_vae_kmpp_history_plotter, vq_vae_ema_history_plotter_eagar, vq_vae_ema_kmpp_history_plotter]
    # print(vq_vae_ema_history_plotter.historys["num_active_embeddings"])
    plot_training_loss(plotter_list)
    
    
def plot_multiple():
    num_models = 10
    
    blue = (68/255, 119/255, 170/255)
    cyan = (102/255, 204/255, 238/255)
    
    green = (34/255, 136/255, 51/255)
    yellow = (204/255, 187/255, 68/255)
    
    red = (238/255, 102/255, 119/255)
    purple = (170/255, 51/255, 119/255)
    
    VQ_VAE_file_list = list()
    VQ_VAE_KMPP_file_list = list()
    VQ_VAE_EMA_file_list = list()
    VQ_VAE_EMA_KMPP_file_list = list()
    VQ_VAE_PCA_file_list = list()
    VQ_VAE_EMA_PCA_file_list = list()
    
    for i in range(num_models):
        VQ_VAE_path = f"training_history_new/VQ_VAE_history/latent_dim_20/num_embeddings_64/dense_vq_vae_input_40_latent_20_num_embeddings_64_init_random_RMSprop_index_{i}.h5"
        VQ_VAE_file_list.append(VQ_VAE_path)
        VQ_VAE_KMPP_path = f"training_history_new/VQ_VAE_KMPP_history/latent_dim_20/num_embeddings_64/dense_vq_vae_input_40_latent_20_num_embeddings_64_init_kmpp_RMSprop_index_{i}.h5"
        VQ_VAE_KMPP_file_list.append(VQ_VAE_KMPP_path)
        VQ_VAE_EMA_path = f"training_history_new/VQ_VAE_EMA_history/latent_dim_20/num_embeddings_64/dense_vq_vae_ema_input_40_latent_20_num_embeddings_64_init_random_RMSprop_ema_decay_0.99_beta_0.25_index_{i}.h5"
        VQ_VAE_EMA_file_list.append(VQ_VAE_EMA_path)
        VQ_VAE_EMA_KMPP_path = f"training_history_new/VQ_VAE_EMA_KMPP_history/latent_dim_20/num_embeddings_64/dense_vq_vae_ema_input_40_latent_20_num_embeddings_64_init_kmpp_RMSprop_ema_decay_0.999_beta_0.25_index_{i}.h5"
        VQ_VAE_EMA_KMPP_file_list.append(VQ_VAE_EMA_KMPP_path)
        VQ_VAE_PCA_path = f"training_history_new/VQ_VAE_PCA_history/latent_dim_20/num_embeddings_64/dense_vq_vae_input_40_latent_20_num_embeddings_64_init_pca_RMSprop_index_{i}.h5"
        VQ_VAE_PCA_file_list.append(VQ_VAE_PCA_path)
        VQ_VAE_EMA_PCA_path = f"training_history_new/VQ_VAE_EMA_PCA_history/latent_dim_20/num_embeddings_64/dense_vq_vae_ema_input_40_latent_20_num_embeddings_64_init_pca_RMSprop_ema_decay_0.99_beta_0.25_index_{i}.h5"
        VQ_VAE_EMA_PCA_file_list.append(VQ_VAE_EMA_PCA_path)
        
    VQ_VAE_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_file_list, label="VQ-VAE-EL", color=blue)
    VQ_VAE_KMPP_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_KMPP_file_list, label="VQ-VAE-EL-KMPP", color=cyan)
    VQ_VAE_EMA_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_EMA_file_list, label="VQ-VAE-EMA", color="g")
    VQ_VAE_EMA_KMPP_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_EMA_KMPP_file_list, label="VQ-VAE-EMA-KMPP", color="r")
    VQ_VAE_PCA_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_PCA_file_list, label="VQ-VAE-EL-PCA", color=red)
    VQ_VAE_EMA_PCA_plotter = TrainingHistoryPlotter_multiple(VQ_VAE_EMA_PCA_file_list, label="VQ-VAE-EMA-PCA", color=purple)
    
    plotter_list = [VQ_VAE_plotter, VQ_VAE_KMPP_plotter, VQ_VAE_PCA_plotter]
    plotter_list_EMA = [VQ_VAE_EMA_plotter, VQ_VAE_EMA_KMPP_plotter, VQ_VAE_EMA_PCA_plotter]
    plotter_list_total = [VQ_VAE_plotter, VQ_VAE_KMPP_plotter, VQ_VAE_PCA_plotter, VQ_VAE_EMA_plotter, VQ_VAE_EMA_KMPP_plotter, VQ_VAE_EMA_PCA_plotter]
    PCA_compare_list = [VQ_VAE_EMA_plotter, VQ_VAE_plotter, VQ_VAE_EMA_PCA_plotter, VQ_VAE_PCA_plotter]
    plot_training_multiple_models(PCA_compare_list)
                        
    
if __name__ == "__main__":
    plot_multiple()
    # single_plot()
    