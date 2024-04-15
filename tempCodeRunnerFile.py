class MyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
        super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l4 = nn.Linear(2048,1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        x = th.unsqueeze(x,1)
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.l2(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.l3(x))
        x = th.flatten(x,start_dim=1)
        x = F.relu(self.l4(x))
        return x