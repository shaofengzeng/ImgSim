
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.sim_net import SimNet
from sim_dataset import SimDataSet
from visdom import Visdom

#vis = Visdom()
#loss_window = vis.line(X=[0],Y=[0])


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):

    max_loss = 9999999.9
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            for i, (base_img, posi_img, nega_img) in tqdm(enumerate(dataloaders[phase])):
                if (base_img is None) or (posi_img is None) or (nega_img is None):
                    continue
                # zero the parameter gradients
                optimizer.zero_grad()

                base_img = base_img.to(device)
                posi_img = posi_img.to(device)
                nega_img = nega_img.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    base_feat = model(base_img)
                    posi_feat = model(posi_img)
                    nega_feat = model(nega_img)
                    loss = criterion(base_feat, posi_feat, nega_feat)
                    #if phase=="train":
                    #    vis.line(X=[epoch*len(dataloaders[phase])+i], Y=[loss.item()],win=loss_window, update='append')
                    print(f"{phase} Loss[{i}/{len(dataloaders[phase])}]: {loss.item():.4f}")

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            # deep copy the model
            if phase == 'val' and epoch_loss < max_loss:
                max_loss = epoch_loss
                torch.save(model.state_dict(), ".\weight\model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_data ={"train":SimDataSet(r".\datasets\train.txt", is_train=True, output_shape=(160, 120)),
               "val":SimDataSet(r".\datasets\test.txt", is_train=False, output_shape=(160, 120))}

    sim_dataloader = {key:DataLoader(sim_data[key], batch_size=2, shuffle=True, collate_fn=sim_data[key].data_collate) \
                      for key in ["train", "val"]}

    criterion = nn.TripletMarginLoss(margin=1.0, p=2., eps=1e-6, reduction="mean")

    model = SimNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5,)
    #optimizer = optim.SGD(model.parameters(),lr=1e-2)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = 5
    train_model(model, sim_dataloader, criterion, optimizer, scheduler, num_epochs, device)




