import torch
import torch.nn as nn
import src.Model as m


def gradient_penalty(critic,real,fake,device="cpu"):
    batch_size,C,H,W = real.shape
    epsilon = torch.randn((batch_size,1,1,1)).repeat(1,C,H,W).to(device)
    intr_images = epsilon * real + (1-epsilon)* fake
    score = critic(intr_images)

    gradiant = torch.autograd.grad(
        inputs=intr_images,
        outputs=score,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradiant = gradiant.view(gradiant.shape[0],-1)
    grad_norm = gradiant.norm(2,dim = 1)
    gradiant_penalty = torch.mean((grad_norm - 1)**2)

    return gradiant_penalty