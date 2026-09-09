from PIL import Image
import torchvision.transforms as transforms

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def preprocess_image(image):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor