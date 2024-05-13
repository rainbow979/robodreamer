


configs = [None for _ in range(100)]

configs[0] = {
    'model_channels': 32,
    'num_res_blocks': 2,
    'attention_resolutions': [8],
    'channel_mult': (1, 2, 4, 8),
    'cross': True,
    'cross_level': [1, 2, 3],
    'multi': True,
    'latent': False,
    'all': False,
    'start_multi': 0,
    'text': 'clip',
    'tiling': True,
    'preload': False,
    'parse_data': True,
}

configs[1] = {
    'model_channels': 128,
    'num_res_blocks': 2,
    'attention_resolutions': [8],
    'channel_mult': (1, 2, 4, 8),
    'cross': True,
    'cross_level': [1, 2, 3],
    'multi': True,
    'latent': False,
    'all': False,
    'start_multi': 40000,
    'text': 'T5',
    'tiling': True,
    'preload': True,
    'parse_data': True,
}

class Config:
    def __init__(self):
        self.config = {
            'model_channels': 128,
            'num_res_blocks': 2,
            'attention_resolutions': [4, 8],
            'channel_mult': (1, 2, 4, 8),
            'cross_level': [2, 3], 
            'cross': True,
            'multi': True,
            'latent': False,
            'all': True,
            'ssr': False,
            'model': 'ego',
            'task_attn': True,
            'comp': True,
            'parse': False,
            'parse_data': False,
            'start_multi': 30000,
            'text': 'T5',
            'tiling': True,
            'preload': False,
            'simple': False,
        }
        self.idx = 0

config = Config()

def init_config(idx):
    if configs[idx] is not None:
        config.config.update(configs[idx])
    config.idx = idx

if __name__ == '__main__':
    init_config(3)
    print(config.config)
