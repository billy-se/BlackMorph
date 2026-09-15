import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

transform = transforms.Compose([#preprocessing pipeline
    transforms.ToTensor(),#translator
    transforms.Normalize((0.1307,), (0.3081,)),#pixel-mean/stdv, 0.1307 is the  brightness of an MNIST handwritting dataset
])#0.3081 is stdv
#so what is MNIST? it is dataset handwritten digits range 0-9, inside has 70k grayscale images of people, size of img is 28x28 pixels
#purpose or where to used? test/training playground for an beginner AI for image recognition or neural network
#so means regular human see it as image but MNIST with AI see it as pure numbers, so in short anything can turn to numbers

train_dataset = torchvision.datasets.MNIST(#turning image to numbers
    root="./data", train=True, download=True, transform=transform #this is the preparation on where the dataset located, if its donwloaded in that folder, then turn them to numbers
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)# divide thems as 64 and shuffle it

class SimpleCNN(nn.Module): #blueprint standard of PyTorch (nn.Module)
    def __init__(self):
        super().__init__() #indicates to run only once
        self.net = nn.Sequential( #self.net is custom variable that hold this entire block, 
            nn.Conv2d(1, 32, 3, 1), # 1=input channels, 32=Output channels filters, 3=each filters scan 9 pixels, 1=1 step size slide so intentionally to repetitive scan of each filters of each pixels
            nn.ReLU(), #ReLU means Reactified Linear Unit which has rules: if number is negative, turn to zero, if its positive, just leave exactly
            nn.MaxPool2d(2), #MaxPool2d means Max Pooling 2-Dimensional which eliminates the intetional repetitive scans in the range of 4 pixels
            nn.Flatten(), #this is most interesting part, earlier raw input was start at 2D then convolution layer turns to 3D (32 stacks 2D grids=3D Tensor) then now .Flatten() means now it turns to 1D
            nn.Linear(32*13*13, 10) # 32=32 visual filters, 13 width=28x28 MNIST image so 28-3 (kernel size) + 1 (comes from natural formula), 13 height=same approach earlier, 10=10 logits within range 0-9
        )
    def forward(self, x): # not my turn
        return self.net(x)

device = torch.device("cpu") #force to use cpu due to limitation of my hardware gpu
model = SimpleCNN().to(device) # run this model with my cpu

criterion = nn.CrossEntropyLoss() #built in fuction that compares model 10 output logits against actual correct label in short 70k+ images batching with 64 -> 5408 -> 10 logits/evaluates
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD=Stochastic Gradient Descent-update the weights, look any errors and tune it, model.parameters=nn.Conv2d or nn.Linear (why OR? both are randomly generating than constantly generating, left subtracts weights, right adding weights), lr=0.01 (Learning Rate means 0.01 reperesent as moderate step size), momentum=0.9, momentum=0.9 means acts like consistency with direction without getting stucked on something

# Modern Opacus syntax
privacy_engine = PrivacyEngine() #comes from Opacus and initialize
model, optimizer, train_loader = privacy_engine.make_private( #grabs model, optimizer and loader then put in a private
    module=model, #
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

epochs = 10 #entire dataset complete/70k MNIST image

for epoch in range(epochs):
    model.train() # neural network is in training mode
    running_loss = 0.0 #tracker of loss or error
    for batch_idx, (data, target) in enumerate(train_loader): # 64 chucks (train_loader), data unpacks the images and target unpacks the batch of corret answer
        data, target = data.to(device), target.to(device) # data and target will go to my cpu then calculates there, what

        optimizer.zero_grad() # erasing gradients (error) to prevent the next datas doing mistakes over and over
        output = model(data) # predicts that range between 0-9 so 10 possible digits/judgement
        loss = criterion(output, target) # it measures model wrong predicts, it compare into real correct answer, wrong = loss score ++ then if right loss score --
        loss.backward() # it figures the mistakes from error, with opacus active, it calculates gradient of every image on batch 64 so image will not LEAK too much and adding noise filter for privacy
        optimizer.step() # model will updates weights and learn so next run will be better than earlier

        running_loss += loss.item() # it traks loss without LEAKING MEMORY

    eps = privacy_engine.get_epsilon(delta=1e-5) # it measures how the model privacy budget progress is
    print( # print for 10x based on epochs
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"ε = {eps:.2f} (δ=1e-5)"
    )

print("Training finished.") # you know already this