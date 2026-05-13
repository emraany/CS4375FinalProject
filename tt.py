from PIL import Image, ImageDraw, ImageFont
import os

files_and_titles = [
    (
        "/Users/emraan/Desktop/mlfinal/figuresToUse/lenetConfusionMatrix.png",
        "LeNet CIFAR-10 Confusion Matrix"
    ),
    (
        "/Users/emraan/Desktop/mlfinal/figuresToUse/seresnetConfusionMatrix.png",
        "SE-ResNet-18 CIFAR-10 Confusion Matrix"
    ),
]

def replace_title(image_path, new_title):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # cover old title area
    draw.rectangle((0, 0, img.width, 60), fill="white")

    # font
    try:
        font = ImageFont.truetype("Arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    # center title
    text_box = draw.textbbox((0, 0), new_title, font=font)
    text_width = text_box[2] - text_box[0]

    x = (img.width - text_width) // 2
    y = 14

    draw.text((x, y), new_title, fill="black", font=font)

    # overwrite same file
    img.save(image_path)
    print(f"Updated: {image_path}")

for image_path, title in files_and_titles:
    replace_title(image_path, title)