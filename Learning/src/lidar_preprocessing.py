from PIL import Image
from math import sin, cos, radians
import torchvision.transforms as transforms

def restrict_lidar_fov(data: list) -> list:
    """IRL Lidar has a 270° FOV. To train the network with data of the same size, 
    restrict the FOV of the lidar data from the simulation. """

    n = len(data)
    angle_increment = 360 / n
    i = round(45 / angle_increment)
    j = n-i-1
    return data[i:j]

def convert_lidar_to_image(data: list) -> Image.Image:
    """See the paper by Mingyu Park et al. for an explanation of what we're doing here. """
    
    def getPos(x):
        """Determines the pixel corresponding to coordinate x"""
        # Consider that (0,0) is the center of the picture, and bound the coordinates
        # within image size. 
        return 112 + max(-111, min(111, round(10*x)))

    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    angle_increment = 270 / len(data)
    theta = -135
    for r in data:
        x = r*sin(radians(theta))
        y = r*cos(radians(theta))
        if r < 2:
            pixels[getPos(x),getPos(y)] = (0,0,round(255/2*r))
        elif r < 4:
            pixels[getPos(x),getPos(y)] = (0,round(255/4*r), 0)
        else:
            pixels[getPos(x),getPos(y)] = (min(255, round(255/6*r)), 0, 0)


        theta += angle_increment

    return img

preprocess_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])