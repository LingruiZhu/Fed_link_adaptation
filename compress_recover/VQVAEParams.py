import sys 
sys.path.append("/home/zhu/Codes/Fed_Link_Adaptation")

class VQVAEParams:
    def __init__(self, autoencoder_type:str, vq_update_type:str, init_embedding_method:str, \
        input_dims:int, latent_dims:int, num_embeddings:int, optimizer:str, num_epochs:int, init_epochs:int, \
            re_init_interval:int, beta:float=0.25, ema_decay:float=0.99, plot_figure:bool=False, \
            num_models:int=None) -> None:
        
        # define model type
        self.autoencoder_type = autoencoder_type
        self.vq_update_type = vq_update_type
        
        # define embedding space
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.num_embeddings = num_embeddings
        self.embedding_init_method = init_embedding_method
        
        # define training parameters
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        
        # for KMPP reinitialization
        self.init_epochs = init_epochs
        self.re_init_interval = re_init_interval
        
        # for EMA algorithm
        self.beta = beta        # beta is the commitment factor
        self.ema_decay = ema_decay
        
        # others
        self.plot_figure = plot_figure
        self.num_models = num_models 