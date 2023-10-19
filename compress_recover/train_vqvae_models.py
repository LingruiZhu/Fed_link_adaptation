import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

from compress_recover.VQVAEParams import VQVAEParams
from compress_recover.vq_vae import train_vq_vae
from compress_recover.vq_vae_ema import train_vq_vae_ema


def train_vqvae_models(params:VQVAEParams):
    if params.vq_update_type == "embedding_loss":
        for index in range(params.num_models):
            train_vq_vae(model_type=params.autoencoder_type,
                        inputs_dims=params.input_dims, 
                        latent_dims=params.latent_dims,
                        num_embeddings=params.num_embeddings,
                        embedding_init=params.embedding_init_method,
                        num_epochs=params.num_epochs,
                        plot_figure=params.plot_figure,
                        optimizer=params.optimizer,
                        init_epochs=params.init_epochs,
                        re_init_interval=params.re_init_interval,
                        simulation_index=index
                        )
            
    elif params.vq_update_type == "ema": 
        for index in range(params.num_models):
            train_vq_vae_ema(model_type=params.autoencoder_type,
                         inputs_dims=params.input_dims,
                         latent_dims=params.latent_dims,
                         num_embeddings=params.num_embeddings,
                         embedding_init=params.embedding_init_method,
                         num_epochs=params.num_epochs,
                         commitment_factor=params.beta,
                         ema_decay=params.ema_dcay,
                         plot_figure=params.plot_figure,
                         optimizer=params.optimizer,
                         init_epochs=params.init_epochs,
                         re_init_interval=params.re_init_interval,
                         simulation_index=index
                         )


def main():
    simulation_parameters = VQVAEParams(
        autoencoder_type="dense",
        vq_update_type="embedding_loss",
        input_dims=40,
        latent_dims=20,
        num_embeddings=64,
        optimizer="RMSprop",
        init_embedding_method="random",
        num_epochs=300,
        init_epochs=100,
        re_init_interval=20,
        beta=0.25,
        ema_decay=0.99,
        plot_figure=False,
        num_models=10,
    )
    train_vqvae_models(simulation_parameters)
    

if __name__ == "__main__":
    main()
        
    