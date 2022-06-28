import torch.nn as nn
from grad_reversal import GradientReversalLayer

class DANNJ(nn.Module):
    
    def __init__(self, num_hidden_layers, num_features):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 256)
            nn.ReLU(True)        
        )
        
        for i in range(num_hidden_layers // 2):
            self.feature_extractor.add_module(
                f'linear_layer_{i + 1}', nn.Linear(256, 256))
            self.feature_extractor.add_module(
                f'linear_layer_{i + 1}', nn.ReLU(True))
            
        self.regressor = nn.Sequential()
        
        for i in range(num_hidden_layers // 2):
            self.regressor.add_module(
                f'linear_layer_{i + 1 + num_hidden_layers // 2}', nn.Linear(256, 256))
            self.regressor.add_module(
                f'linear_layer_{i + 1 + num_hidden_layers // 2}', nn.ReLU(True))
            
        self.regressor.add_module(
            f'regressor_output', nn.Linear(256, 1)
        )
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_linear_layer_1', nn.Linear(256, 256))
        self.domain_classifier.add_module('dc_batch_norm_1', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('dc_relu_1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_linear_layer_2', nn.Linear(256, 256))
        self.domain_classifier.add_module('dc_batch_norm_2', nn.BatchNorm1d(256))
        self.domain_classifier.add_module('dc_relu_2', nn.ReLU(True))
        self.domain_classifier.add_module('dc_linear_layer_3', nn.Linear(256s, 2))
        self.domain_classifier.add_module('dc_softmas', nn.LogSoftmax(dim=1))
        
    def forward(self, input_data, grl_lam=0.0):
        
        features = self.feature_extractor(input_data)
        features = feature.view(-1, 256)
        features_grl = GradientReversalLayer.apply(features, grl_lam)
        regressor_pred = self.regressor(features)
        domain_pred = self.domain_classifier(features_grl)

        return regressor_pred, domain_pred

