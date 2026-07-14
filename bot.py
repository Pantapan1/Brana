import asyncio
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from model import Generator, Discriminator, TextEncoder

TOKEN = "8656185873:AAEuggUpzeDNDZv4jtX7OkzFyey0KTLH1Tg"
DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Словарь для токенизации (простой словарь слов)
vocab = {"<pad>": 0, "<unk>": 1}

def tokenize(text):
    tokens = []
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens.append(vocab[word])
    # Добиваем паддингом до фиксированной длины (например, 10 слов)
    while len(tokens) < 10:
        tokens.append(0)
    return tokens[:10]

# Подготовка ИИ моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = TextEncoder(vocab_size=5000).to(device)
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Бот-ГАН готов. Скидывай фото с описанием в caption. Команда /train обучит ИИ, а /gen <текст> — создаст арт.")

@dp.message(F.photo)
async def collect(message: types.Message):
    if not message.caption:
        return await message.answer("Добавь описание к картинке!")
    
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_id = len(os.listdir(IMG_DIR))
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    await bot.download_file(file.file_path, img_path)
    
    with open(os.path.join(DATA_DIR, "labels.txt"), "a", encoding="utf-8") as f:
        f.write(f"{img_id}.jpg|{message.caption}\n")
        
    await message.answer(f"Загружено! Общий датасет: {img_id + 1} фото.")

@dp.message(Command("train"))
async def train(message: types.Message):
    label_path = os.path.join(DATA_DIR, "labels.txt")
    if not os.path.exists(label_path):
        return await message.answer("Датасет пуст!")
        
    await message.answer("⚙️ Обучаю ИИ на текущем датасете (1 эпоха)...")
    
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        img_name, caption = line.strip().split("|")
        img = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
        real_imgs = transform(img).unsqueeze(0).to(device)
        
        tokens = torch.tensor([tokenize(caption)]).to(device)
        text_embed = encoder(tokens)
        
        batch_size = real_imgs.size(0)
        label_real = torch.ones(batch_size, 1).to(device)
        label_fake = torch.zeros(batch_size, 1).to(device)
        
        # --- Тренируем Дискриминатор ---
        optimizerD.zero_grad()
        output_real = netD(real_imgs, text_embed)
        lossD_real = criterion(output_real, label_real)
        
        noise = torch.randn(batch_size, 100).to(device)
        fake_imgs = netG(noise, text_embed)
        output_fake = netD(fake_imgs.detach(), text_embed)
        lossD_fake = criterion(output_fake, label_fake)
        
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()
        
        # --- Тренируем Генератор ---
        optimizerG.zero_grad()
        output = netD(fake_imgs, text_embed)
        lossG = criterion(output, label_real)
        lossG.backward()
        optimizerG.step()
        
    await message.answer("✅ Веса обновлены. ИИ сделал шаг к пониманию твоих артов.")

@dp.message(Command("gen"))
async def generate(message: types.Message):
    prompt = message.text.replace("/gen", "").strip()
    if not prompt:
        return await message.answer("Пример: /gen красный меч")
        
    tokens = torch.tensor([tokenize(prompt)]).to(device)
    with torch.no_grad():
        text_embed = encoder(tokens)
        noise = torch.randn(1, 100).to(device)
        fake_img = netG(noise, text_embed).cpu().squeeze(0)
        
    # Денормализация тензора в картинку
    fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(fake_img)
    img.save("out.jpg")
    
    await message.answer_photo(types.FSInputFile("out.jpg"), caption=f"Генерация по запросу: {prompt}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
