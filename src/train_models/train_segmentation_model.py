import os
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from dataset.dataloader import get_segmentation_dataloader


def save_model(model, path="../models/unet_resnet34_segmentation.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def build_model(device):
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                num_epochs=5):
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # you could also run a validation pass here
        print(f"Epoch {epoch}/{num_epochs} — train loss: {train_loss:.4f}")


if __name__ == "__main__":
    # choose your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloaders (pass your paths here)
    train_loader, val_loader = get_segmentation_dataloader(
        image_dir="../../data/skin_cancer_data/images",
        mask_dir="../../data/skin_cancer_data/masks"
    )

    # build model, loss, optimizer
    model = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # run training
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=5
    )

    save_model(model)