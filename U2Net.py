import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dirate=1):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1*dirate,dilation=1*dirate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, inputs):
        return self.model(inputs)
    
class MINI_Unet(nn.Module):
    def __init__(self, sampling_layers = 5, in_ch=3, mid_ch=12, out_ch=3):
        super(MINI_Unet, self).__init__()
        self.input1 = ConvBlock(in_ch, out_ch)
        self.input2 = ConvBlock(out_ch, mid_ch)

        self.downs = nn.ModuleList(
            [
                ConvBlock(mid_ch, mid_ch)
                for _ in range(sampling_layers)
            ]
        )
        self.ups = nn.ModuleList(
            [
               ConvBlock(mid_ch*2, mid_ch)
               for _ in range(sampling_layers)
            ]
        )
        self.mid1 = ConvBlock(mid_ch, mid_ch, dirate=2)
        self.mid2 = ConvBlock(mid_ch, mid_ch, dirate=1)
        self.output = ConvBlock(mid_ch*2, out_ch)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    
    def forward(self, inputs):
        skips = []

        x1 = self.input1(inputs)
        x = x1

        x = self.input2(x)
        skips.append(x)

        for down in self.downs:
            conv = down(x)
            skips.append(conv)
            x = self.pool(conv)
        
        x = self.mid1(x)
        x = self.mid2(x)

        for up in self.ups:
            conv = skips.pop()
            x = up(torch.cat((_upsample_like(x,conv), conv), 1))

        x = self.output(torch.cat((x, skips.pop()), 1))

        return x + x1


class MINI_Unet2(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(MINI_Unet2, self).__init__()
        self.input1 = ConvBlock(in_ch, out_ch)
        self.input2 = ConvBlock(out_ch, mid_ch)

        self.l1 = ConvBlock(mid_ch, mid_ch, dirate=2)    
        self.l2 = ConvBlock(mid_ch, mid_ch, dirate=4)    

        self.mid = ConvBlock(mid_ch, mid_ch, dirate=8)

        self.r1 = ConvBlock(mid_ch*2, mid_ch, dirate=4)    
        self.r2 = ConvBlock(mid_ch*2, mid_ch, dirate=2)    
    
        self.output = ConvBlock(mid_ch*2, out_ch)

    
    def forward(self, inputs):
        skips = []

        x1 = self.input1(inputs)
        x = x1

        x = self.input2(x)
        skips.append(x)

        x = self.l1(x)
        skips.append(x)

        x = self.l2(x)
        skips.append(x)

        x = self.mid(x)
        
        x = self.r1(torch.cat((x, skips.pop()), 1))
        x = self.r2(torch.cat((x, skips.pop()), 1))

        x = self.output(torch.cat((x, skips.pop()), 1))

        return x + x1

def _upsample_like(src,tar):
        src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
        return src


class U2Net_Light(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2Net_Light, self).__init__()
        self.stage1 = MINI_Unet(5,in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = MINI_Unet(4,64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = MINI_Unet(3,64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = MINI_Unet(2,64,16,32)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = MINI_Unet2(32,16,16)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = MINI_Unet2(16,8,16)

        # decoder
        self.stage5d = MINI_Unet2(32,16,32)
        self.stage4d = MINI_Unet(2,64,16,64)
        self.stage3d = MINI_Unet(3,128,16,64)
        self.stage2d = MINI_Unet(4,128,16,64)
        self.stage1d = MINI_Unet(5,128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(32,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(16,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        s = hx6.shape
        hx6 = hx6.view(hx6.size(0), -1)
        hx6 = hx6.view(s)

        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
