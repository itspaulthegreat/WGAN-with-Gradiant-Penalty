import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import src
import src.Model as m
import src.utils as u
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#define variables
epochs = 5
img_channels = 3 #for Mnist 1 and for color 3
batch_size = 64
img_size = 64
lr = 1e-4
features_c = 64
features_g = 64
z_dim = 100
cric_epochs = 5
lambda_gp = 10
#transforms
transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)] , [0.5 for _ in range(img_channels)])
])

#dataset
#dataset = datasets.MNIST(root = "dataset/",download=True,transform= transform)
dataset = datasets.ImageFolder(root = "celeb_dataset",transform= transform)

loader = DataLoader(dataset,batch_size=batch_size,shuffle=True )
 

#create objects
gen = m.Generator(z_dim,img_channels,features_g).to(device)
cric = m.Critic(img_channels,features_c).to(device)

#initialize weight
m.initialize_weights(cric)
m.initialize_weights(gen)

#optimizer
opti_cric = optim.Adam(cric.parameters(),lr = lr,betas=(0.0,0.9))
opti_gen = optim.Adam(gen.parameters(),lr = lr,betas=(0.0,0.9))

#fixed noise

fixed_noise = torch.randn((batch_size,z_dim,1,1))

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

cric.train()
gen.train()

for epoch in range(epochs):
    for batch_idx , (real,_) in enumerate(loader):
        real = real.to(device)
        
        for _ in range(cric_epochs):
            noise = torch.randn((batch_size,z_dim,1,1)).to(device)
            cric_real = cric(real).reshape(-1)
            fake = gen(noise) 
            cric_fake = cric(fake).reshape(-1)
            gradient_penalty = u.gradient_penalty(cric,real,fake,device=device)
            loss_cric = -(torch.mean(cric_real) - torch.mean(cric_fake)) + (lambda_gp * gradient_penalty)
            cric.zero_grad()
            loss_cric.backward(retain_graph = True)
            opti_cric.step()


        output = cric(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opti_gen.step()
    
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_cric:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "FAKE", img_grid_fake, global_step = step
                )

                writer_real.add_image(
                    "REAL", img_grid_real, global_step = step
                )

                step += 1