import torch

import torch.nn as nn 


def crop_image(original, target):
    target_size = target.size()[2]
    original_size = original.size()[2]
    delta  = original_size - target_size
    delta  = delta // 2
    return original[:, :, delta:original_size-delta,  delta:original_size-delta]



def double_conv_layer(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(
        in_channels= in_c, 
        out_channels= out_c,  
        kernel_size = 3),

        nn.ReLU(inplace=True),
        

        nn.Conv2d(in_channels= out_c,
        out_channels=out_c,
        kernel_size=3),

        nn.ReLU(inplace=True)
        
    )

    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride= 2)

        self.down_conv_1 = double_conv_layer(1, 64)
        self.down_conv_2 = double_conv_layer(64, 128)
        self.down_conv_3 = double_conv_layer(128, 256)
        self.down_conv_4 = double_conv_layer(256, 512)
        self.down_conv_5 = double_conv_layer(512, 1024)

        self.up_trans_1  = nn.ConvTranspose2d(
            in_channels= 1024,
            out_channels= 512,
            kernel_size= 2,
            stride= 2)
        self.up_conv_1 = double_conv_layer(1024, 512)
        
        self.up_trans_2  = nn.ConvTranspose2d(
            in_channels= 512,
            out_channels= 256,
            kernel_size= 2,
            stride= 2)
        self.up_conv_2 = double_conv_layer(512, 256)
        
        self.up_trans_3  = nn.ConvTranspose2d(
            in_channels= 256,
            out_channels= 128,
            kernel_size= 2,
            stride= 2)
        self.up_conv_3 = double_conv_layer(256, 128)
        
        self.up_trans_4  = nn.ConvTranspose2d(
            in_channels= 128,
            out_channels= 64,
            kernel_size= 2,
            stride= 2)
        self.up_conv_4 = double_conv_layer(128, 64)

        self.out = nn.Conv2d(
            in_channels= 64,
            out_channels= 2,
            kernel_size= 1
        )
        


        

    def forward(self, image):
        # encoder bs c h w
        
        print ("--------------------encoding part--------------------")


        x1 = self.down_conv_1(image)
        p1 = self.max_pool_2x2(x1)
        print (x1.size())
        print ('\n')

        x2 = self.down_conv_2(p1)
        p2 = self.max_pool_2x2(x2)
        print (x2.size())
        print ('\n')
 
        x3 = self.down_conv_3(p2)
        p3 = self.max_pool_2x2(x3)
        print (x3.size())
        print ('\n')

        x4 = self.down_conv_4(p3)
        p4 = self.max_pool_2x2(x4)
        print (x4.size())
        print ('\n')

        x5 = self.down_conv_5(p4)

        print (x5.size())
        print ('\n')

        # decode 
        print ("----------------------decoding part-------------------")

        x = self.up_trans_1(x5)
        y = crop_image(x4, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        print (x.size())
        print ('\n')

        x = self.up_trans_2(x)
        y = crop_image(x3, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        print (x.size())
        print ('\n')

        x = self.up_trans_3(x)
        y = crop_image(x2, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        print (x.size())
        print ('\n')

        x = self.up_trans_4(x)
        y = crop_image(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        print (x.size())
        print ('\n')
        
        # adding output layer
        print("----------------------output layer-------------------")
        x = self.out(x)
        print (x.size())
        print ('\n')

        return x





if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print (model(image))



         
        



