import torch
import torch.nn as nn

# Простейший кодировщик текста: переводит индексы слов в вектор
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, text_tokens):
        # Усредняем векторы всех слов в предложении
        return self.embedding(text_tokens).mean(dim=1)

# Генератор: Шум (100) + Текст (100) -> Картинка (3, 64, 64)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, embed_dim=100):
        super().__init__()
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embed_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_embed):
        gen_input = torch.cat((noise, text_embed), dim=1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Дискриминатор: Картинка (3, 64, 64) + Текст (100) -> Реально/Фейк (1)
class Discriminator(nn.Module):
    def __init__(self, embed_dim=100):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Проекция текста под размер картинки
        self.text_layer = nn.Linear(embed_dim, 64 * 64)
        
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1), # 4 канала: 3 (RGB) + 1 (текст)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_embed):
        # Разворачиваем эмбеддинг текста в матрицу 64x64 и добавляем как 4-й канал к фото
        t_layer = self.text_layer(text_embed).view(-1, 1, 64, 64)
        d_in = torch.cat((img, t_layer), dim=1)
        validity = self.model(d_in)
        return validity
