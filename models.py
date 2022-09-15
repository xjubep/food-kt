import timm
from torch import nn


class ImageModel(nn.Module):
    def __init__(self, model_name, class_n, mode='train'):
        super().__init__()
        self.model_name = model_name.lower()
        self.class_n = class_n
        self.mode = mode
        self.encoder = timm.create_model(self.model_name, pretrained=False)

        names = []
        modules = []
        for name, module in self.encoder.named_modules():
            names.append(name)
            modules.append(module)

        self.fc_in_features = self.encoder.num_features
        print(f'The layer was modified...')

        fc_name = names[-1].split('.')

        if len(fc_name) == 1:
            print(
                f'{getattr(self.encoder, fc_name[0])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(self.encoder, fc_name[0], nn.Linear(self.fc_in_features, class_n))
        elif len(fc_name) == 2:
            print(
                f'{getattr(getattr(self.encoder, fc_name[0]), fc_name[1])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(getattr(self.encoder, fc_name[0]), fc_name[1], nn.Linear(self.fc_in_features, class_n))

    def forward(self, x):
        x = self.encoder(x)
        return x
