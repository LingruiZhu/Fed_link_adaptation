class VQVAEParams:
    def __init__(self, autoencoder_type:str, vq_update_type:str, init_method:str, \
        input_dims:int, latent_dims:int, optimizer:str, num_epochs:int) -> None:
        self.autoencoder_type = autoencoder_type
        self.vq_update_type = vq_update_type
        self.init_method = init_method
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.optimizer = optimizer
        self.num_epochs = num_epochs