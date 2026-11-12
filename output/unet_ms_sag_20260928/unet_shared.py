"""Shared Wang-inspired UNet; explicit adaptation, not author source."""
import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self,a,b,drop=False):
        modules=[nn.Conv2d(a,b,3,padding=1,padding_mode='replicate'),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,padding_mode='replicate'),nn.ReLU()]
        if drop:modules.append(nn.Dropout(.5))
        super().__init__(*modules)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.ModuleList([Block(3,64),Block(64,128),Block(128,256),Block(256,512,True),Block(512,1024,True)])
        self.pool=nn.MaxPool2d(2)
        self.up=nn.ModuleList([nn.ConvTranspose2d(a,b,2,stride=2) for a,b in [(1024,512),(512,256),(256,128),(128,64)]])
        self.dec=nn.ModuleList([Block(a,b) for a,b in [(1024,512),(512,256),(256,128),(128,64)]])
        self.head=nn.Sequential(nn.Conv2d(64,2,3,padding=1,padding_mode='replicate'),nn.ReLU(),nn.Conv2d(2,1,1))
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:nn.init.zeros_(m.bias)
    def forward(self,x):
        skips=[]
        for i,enc in enumerate(self.enc):
            x=enc(x);skips.append(x)
            if i<4:x=self.pool(x)
        for up,dec,skip in zip(self.up,self.dec,reversed(skips[:-1])):x=dec(torch.cat([up(x),skip],dim=1))
        return self.head(x)

def weighted_loss(logits,target):
    """Eq.4-5: foreground coefficient1; background coefficient fg/bg, per image."""
    fg=target.sum((1,2,3),keepdim=True); bg=target[0].numel()-fg
    if torch.any(fg<=0) or torch.any(bg<=0):raise ValueError('Empty/full labels need a separate, explicit protocol')
    weights=target+(1-target)*fg/bg
    return (nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).mean()
