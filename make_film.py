import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
path_to_images = 'C:/Users/js/Documents/P3Results/Frames/'
path_to_videos = 'C:/Users/js/Documents/P3Results/'

image_files = []

for img_number in range(1,2414): 
    image_files.append(path_to_images + str(img_number) + '.jpg') 

fps = 25
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path_to_videos + 'my_new_video.mp4')