import os
import glob
from moviepy.editor import ImageSequenceClip


def make_video(path, video_name, fps=30): 
    images = sorted(glob.glob(os.path.join(path, '*.png')))  
    print('Making video from', len(images), 'images')
    clip = ImageSequenceClip(images, fps=fps)
    os.makedirs(os.path.dirname(video_name), exist_ok=True)
    clip.write_videofile(video_name)


if __name__ == '__main__': 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing image sequences')
    parser.add_argument('--output', type=str, required=False, help='Path to the output folder for saving videos')
    parser.add_argument('--group', action='store_true', help='Convert all subfolders in the input folder to mp4')
    args = parser.parse_args()

    input_folder = args.input
    if args.group:
        subfolders = glob.glob(os.path.join(input_folder, '*/'))
        for subfolder in subfolders:
            print('Making video from', subfolder)
            output_file = subfolder[:-1] + '.mp4'
            make_video(subfolder, output_file, fps=30)
    else:
        output_folder = args.output
        if output_folder is None:
            output_folder = input_folder[:-1] if input_folder[-1] == '/' else input_folder
            output_folder += '.mp4'
            print('Output folder not specified, using %s' % output_folder)
        make_video(input_folder, output_folder, fps=30)
