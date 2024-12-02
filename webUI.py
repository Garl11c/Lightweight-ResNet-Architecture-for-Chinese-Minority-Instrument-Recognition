import os
import torch
import torchaudio
from torchvision.models.resnet import ResNet, BasicBlock
import gradio as gr


INSTRUMENT_DESCRIPTIONS = {
    'Ghaychak': 'The Ghaychak is a traditional bowed string instrument popular in Central Asia.',
    'Dutar': 'The Dutar is a two-stringed lute, widely used in the folk music of Central Asia.',
    'Husital': 'Husital is a unique wind instrument known for its soft and resonant tone.',
    'Sattar': 'The Sattar is a traditional Uyghur instrument with multiple strings, often used in classical performances.',
    'Daf': 'The Daf is a frame drum commonly used in Sufi music and traditional celebrations.',
    'Bombur': 'Bombur is a traditional plucked string instrument known for its deep and mellow sound.'
}

# 定义模型和加载
class ResNet10(ResNet):
    def __init__(self, num_classes):
        super(ResNet10, self).__init__(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            num_classes=num_classes
        )

class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.resnet = ResNet10(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_audio(file_path, resize=224):
    waveform, sample_rate = torchaudio.load(file_path)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=640,
        n_mels=128
    )
    mel_spectrogram = transform(waveform)
    mel_spectrogram = torch.clamp(mel_spectrogram, min=0)
    mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
    resize_transform = torch.nn.AdaptiveAvgPool2d((resize, resize))
    mel_spectrogram = resize_transform(mel_spectrogram)
    return mel_spectrogram.unsqueeze(0)

def classify_audio(model, file_path, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_spectrogram = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
        _, predicted = outputs.max(1)
        return labels[predicted.item()]

# 加载模型和标签
NUM_CLASSES = 6
LABELS = ['Ghaychak', 'Dutar', 'Husital', 'Sattar', 'Daf', 'Bombur']
MODEL_PATH = 'final_audio_classifier.pth'
model = load_model(MODEL_PATH, num_classes=NUM_CLASSES)

def predict(audio_path):
    # 获取预测类别
    predicted_class = classify_audio(model, audio_path, LABELS)
    # 设置图片路径
    image_path = os.path.join("images", f"{predicted_class}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image for class {predicted_class} not found at {image_path}.")
    # 获取乐器英文介绍
    description = INSTRUMENT_DESCRIPTIONS.get(predicted_class, "No description available.")
    # 返回类别、图片路径和英文介绍
    return predicted_class, image_path, description



interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload an audio file"),
    outputs=[
        gr.Label(label="Predicted Class"),  # 输出类别
        gr.Image(label="Category Image"),  # 输出图片
        gr.Textbox(label="Instrument Description")  # 输出乐器介绍
    ],
    title="Audio Classification with ResNet",
    description="Upload an audio file to classify it into one of the predefined categories and see its corresponding image and description."
)


if __name__ == "__main__":
    interface.launch()
