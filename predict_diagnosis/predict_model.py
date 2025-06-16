import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# 모델 클래스 정의 (MODEL.ipynb에서 추출)
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [batch, seq_len]
        x = self.embedding(x)           # [batch, seq_len, embed_dim]
        x = x.permute(1, 0, 2)          # [seq_len, batch, embed_dim]
        out = self.encoder(x)           # [seq_len, batch, embed_dim]
        out = out.mean(dim=0)           # mean pooling
        return self.fc(out)             # [batch, num_classes]

# ATC 코드 사전과 질병 라벨 매핑 파일 불러오기
with open("atc_vocab.json", "r", encoding='utf-8') as f:
    atc_to_index = json.load(f)

with open("label_map.json", "r", encoding='utf-8') as f:
    index_to_label = json.load(f)

# ATC 코드 → 텐서로 변환
def encode_atc_codes(atc_list, max_len=50):
    encoded = [atc_to_index.get(code, atc_to_index.get("<UNK>", 1)) for code in atc_list]
    if len(encoded) < max_len:
        encoded += [atc_to_index.get("<PAD>", 0)] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return torch.tensor([encoded], dtype=torch.long)

# 모델 로딩
def load_model(model_path, device='cpu'):
    # 고정된 vocab_size 사용 (학습 당시 vocab 사전 크기 - 1)
    vocab_size = 151  # 정확한 사이즈를 지정해야 함 (추출된 json 말고 학습 당시 기준)
    num_classes = len(index_to_label)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        embed_dim=32,
        nhead=4,
        num_layers=2,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 예측 함수
def predict_diagnosis(model, atc_list, device='cpu'):
    input_tensor = encode_atc_codes(atc_list).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return index_to_label[str(pred_class)], probs.squeeze().tolist()

# 사용 예시
if __name__ == '__main__':
    atc_input = ["A02BC01", "C09AA05", "N06AB10"]  # 사용자 입력
    model = load_model("trained_transformer_model.pth", device="cpu")
    diagnosis, prob = predict_diagnosis(model, atc_input)
    print(f"예측 질병명: {diagnosis}")