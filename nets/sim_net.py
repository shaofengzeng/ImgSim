import torch
import torch.nn as nn
import torchvision
    
class SimNet(nn.Module):
    def __init__(self, ):
        super(SimNet, self).__init__()
        self.vgg = torchvision.models.vgg11(pretrained=True)
        #self.vgg = torchvision.models.resnet18(pretrained=True)
        #self.vgg = torchvision.models.vgg11_bn(pretrained=True)
        #self.vgg = torchvision.models.vgg16(pretrained=True)
        #self.vgg = torchvision.models.mobilenet_v2(pretrained=True)
        #self.vgg = torchvision.models.alexnet()

        self.conv_feat = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.fc_feat = nn.Sequential(
            nn.Linear(256*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )

        #init weight
        w_dict = {'0.weight':'0.weight', '0.bias':'0.bias',\
                 '3.weight':'3.weight', '3.bias':'3.bias',\
                 '6.weight':'6.weight', '6.bias':'6.bias',\
                 '9.weight':'8.weight', '9.bias':'8.bias'}
        vgg16_feat_state_dict = self.vgg.features.state_dict()
        conv_feat_state_dict = self.conv_feat.state_dict()
        for k,v in w_dict.items():
            conv_feat_state_dict[k] = vgg16_feat_state_dict[v]
        self.conv_feat.load_state_dict(conv_feat_state_dict)
        #冻结权重更新
        for param in self.conv_feat.parameters():
            param.requires_grad = False

    def forward(self, data):
        x = self.conv_feat(data)
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = self.fc_feat(x)

        return x

if __name__=="__main__":
    import random
    sim = SimNet()
    for i in range(100):
        w = random.randint(224, 224)
        h = random.randint(224, 224)
        data = torch.rand([2,3,h,w])
        d = sim(data)