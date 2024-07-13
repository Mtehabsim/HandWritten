import os
import random

def generate_16_digit_int():
    return random.randint(10**15, 10**16 - 1)

def rename_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            new_filename = f"{generate_16_digit_int()}{os.path.splitext(filename)[1]}"
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            
image_directory = r'D:\cloud\OneDrive - AL-Hussien bin Abdullah Technical University\Desktop\PycharmProjects\htrCNN\HandWritten\uploaded_images'
rename_images(image_directory)
