## BACKGROUND REMOVAL USING U2NET PRETRAINED MODEL ##

orginal repo: https://github.com/xuebinqin/U-2-Net

- pretrained model obtained from https://github.com/renatoviolin/bg-remove-augment

steps to build docker containers: 

- pip install gdown 

- gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O ./ckpt/u2net.pth

- download the model and place it in the folder backend

- From the root directory run docker-compose up -d --build
