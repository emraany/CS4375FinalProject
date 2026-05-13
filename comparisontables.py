from PIL import Image, ImageDraw, ImageFont
import os

RESULTS_DIR = "results"

def add_title(image, title):
    image = image.convert("RGB")

    title_height = 55
    new_image = Image.new("RGB", (image.width, image.height + title_height), "white")
    new_image.paste(image, (0, title_height))

    draw = ImageDraw.Draw(new_image)

    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    text_box = draw.textbbox((0, 0), title, font=font)
    text_width = text_box[2] - text_box[0]

    x = (image.width - text_width) // 2
    y = 12

    draw.text((x, y), title, fill="black", font=font)

    return new_image


def combine_images_with_titles(image_info, output_path):
    titled_images = []

    for path, title in image_info:
        img = Image.open(path)
        img = add_title(img, title)
        titled_images.append(img)

    # resize all images to same height
    target_height = min(img.height for img in titled_images)

    resized = []
    for img in titled_images:
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        resized.append(img.resize((new_width, target_height)))

    total_width = sum(img.width for img in resized)
    combined = Image.new("RGB", (total_width, target_height), "white")

    x_offset = 0
    for img in resized:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    combined.save(output_path)
    print(f"Saved: {output_path}")


# MNIST accuracy
combine_images_with_titles(
    [
        (os.path.join(RESULTS_DIR, "lenet_mnist", "accuracy_curve.png"), "LeNet on MNIST"),
        (os.path.join(RESULTS_DIR, "resnet18_mnist", "accuracy_curve.png"), "ResNet-18 on MNIST"),
        (os.path.join(RESULTS_DIR, "se_resnet18_mnist", "accuracy_curve.png"), "SE-ResNet-18 on MNIST"),
    ],
    os.path.join(RESULTS_DIR, "mnist_accuracy_comparison.png")
)

# CIFAR-10 accuracy
combine_images_with_titles(
    [
        (os.path.join(RESULTS_DIR, "lenet_cifar10", "accuracy_curve.png"), "LeNet on CIFAR-10"),
        (os.path.join(RESULTS_DIR, "resnet18_cifar10", "accuracy_curve.png"), "ResNet-18 on CIFAR-10"),
        (os.path.join(RESULTS_DIR, "se_resnet18_cifar10", "accuracy_curve.png"), "SE-ResNet-18 on CIFAR-10"),
    ],
    os.path.join(RESULTS_DIR, "cifar10_accuracy_comparison.png")
)

# MNIST loss
combine_images_with_titles(
    [
        (os.path.join(RESULTS_DIR, "lenet_mnist", "loss_curve.png"), "LeNet on MNIST"),
        (os.path.join(RESULTS_DIR, "resnet18_mnist", "loss_curve.png"), "ResNet-18 on MNIST"),
        (os.path.join(RESULTS_DIR, "se_resnet18_mnist", "loss_curve.png"), "SE-ResNet-18 on MNIST"),
    ],
    os.path.join(RESULTS_DIR, "mnist_loss_comparison.png")
)

# CIFAR-10 loss
combine_images_with_titles(
    [
        (os.path.join(RESULTS_DIR, "lenet_cifar10", "loss_curve.png"), "LeNet on CIFAR-10"),
        (os.path.join(RESULTS_DIR, "resnet18_cifar10", "loss_curve.png"), "ResNet-18 on CIFAR-10"),
        (os.path.join(RESULTS_DIR, "se_resnet18_cifar10", "loss_curve.png"), "SE-ResNet-18 on CIFAR-10"),
    ],
    os.path.join(RESULTS_DIR, "cifar10_loss_comparison.png")
)