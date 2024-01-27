import torch
import torchvision

from network import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model = MSTHNet_resnet18_final(23)
# model.load_state_dict(torch.load('results/MSTHNet_resnet18_final_minc_split1_06211710.pt'))

# model = resnet18(pretrained=True)
# model.fc = nn.Linear(512,23)
# model = nn.DataParallel(model)
# model.to(device)

# model.load_state_dict(torch.load('results/ResNet_minc_split1_06230000.pt'))
#model = torch.load('/home/percv-d10/git/han/EBLNet/best_epoch_17_mean-iu_0.89302.pth')#.to('cpu')
model = torch.load('/home/percv-d10/git/han/EBLNet/best_epoch_17_mean-iu_0.89302.pth', map_location=torch.device('cpu'))
example = torch.rand((1, 3, 512, 512))
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("model_resnet_jit_cuda.pt")