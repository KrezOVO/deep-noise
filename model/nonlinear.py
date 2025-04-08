import torch
from torch import nn

class NonLinear(nn.Module):
    def __init__(self, in_nc=6, nc=1600, out_nc=1, num_sheets=4):
        super(NonLinear, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_sheets)
        self.num_sheets = num_sheets
        self.out_nc = out_nc

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        return output
    

class NonLinearType(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18):
        super(NonLinearType, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)


    def forward(self, inp, type_):
        type_ = torch.LongTensor([type_])
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output
    
class NonLinearBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, num_bins=25):
        super(NonLinearBin, self).__init__()
        self.num_bins = num_bins
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, num_bins)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        return output

class NonLinearTypeBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_bins=51):
        super(NonLinearTypeBin, self).__init__()
        self.num_bins = num_bins
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc*num_bins)

    def forward(self, inp, type_):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output
    
class NonLinearMultiBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=1):
        super(NonLinearMultiBin, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)
        self.out0 = nn.Linear(nc, 18)
        self.out1 = nn.Linear(nc, 6)
        self.out2 = nn.Linear(nc, 4)


    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        output0 = self.out0(x)
        output1 = self.out1(x)
        output2 = self.out2(x)
        return output, output0, output1, output2

class NonLinearTypeBinModel(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_bins=26, num_sheets=4):
        super(NonLinearTypeBinModel, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_bins * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc, self.num_bins)
        return output

class NonLinearBinModel(nn.Module):
    def __init__(self, in_nc=6, nc=1600, num_bins=26, num_sheets=4):
        super(NonLinearBinModel, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, num_sheets * num_bins)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.num_bins)
        return output

class NonLinearTypeModel(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_sheets=4):
        super(NonLinearTypeModel, self).__init__()
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc)
        return output

class NonLinearBowlBinMode(nn.Module):
    def __init__(self, in_nc=6, nc=1600, out_nc=2, num_bins=26, num_sheets=4):
        super(NonLinearBowlBinMode, self).__init__()
        self.num_bins = num_bins
        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_bins * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc, self.num_bins)
        return output

class NonLinearBowlMode(nn.Module):
    def __init__(self, in_nc=6, nc=1600, out_nc=2, num_sheets=4):
        super(NonLinearBowlMode, self).__init__()

        self.num_sheets = num_sheets
        self.out_nc = out_nc
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc * num_sheets)

    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.view(-1, self.num_sheets, self.out_nc)
        return output