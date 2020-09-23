import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "vgg19": models.vgg19(pretrained=False)
                            }

        model = self._get_basemodel(base_model)

        try:
            num_ftrs = model.fc.in_features
        except:
            num_ftrs = model.classifier[0].in_features

        self.features = nn.Sequential(*list(model.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, 2048)
        self.l2 = nn.Linear(2048, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, 1)

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
