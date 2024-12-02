import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import swanlab
from torchvision.models.resnet import ResNet, BasicBlock

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, df, resize, train_mode=True):
        self.audio_paths = df['file_path'].values
        self.labels = df['label'].values
        self.resize = resize
        self.train_mode = train_mode

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=640,
            n_mels=128
        )
        mel_spectrogram = transform(waveform)
        mel_spectrogram = torch.clamp(mel_spectrogram, min=0)
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
        resize = torch.nn.AdaptiveAvgPool2d((self.resize, self.resize))
        mel_spectrogram = resize(mel_spectrogram)
        return mel_spectrogram, self.labels[idx]



class ResNet10(ResNet):
    def __init__(self, num_classes):
        super(ResNet10, self).__init__(
            block=BasicBlock,
            layers=[1, 1, 1, 1],  # 每个阶段只有 1 个 BasicBlock
            num_classes=num_classes
        )
# 修改ResNet模型


class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.resnet = ResNet10(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)


def save_model(model, path="audio_classifier.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "train/epoch": epoch,
            "train/lr": current_lr
        })
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = swanlab.init(
        project="Audio_Classification_with_ResNet18",
        experiment_name="resnet18_audio",
        config={
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 20,
            "resize": 224,
        },
    )
    df = pd.read_csv("audio_labels.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = AudioDataset(train_df, resize=run.config["resize"], train_mode=True)
    val_dataset = AudioDataset(val_df, resize=run.config["resize"], train_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=run.config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=run.config["batch_size"], shuffle=False)
    num_classes = len(df['label'].unique())
    model = AudioClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config["learning_rate"])
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=run.config["num_epochs"], device=device)
    save_model(model, path="final_audio_classifier.pth")


if __name__ == "__main__":
    main()
