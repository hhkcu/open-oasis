# @title Parse Starting Images
import os, pathlib, ffmpeg, sys

crop_width=640
crop_height=360

imagesPath = "images"
samplesPath = "open-oasis/sample-data"
for fn in os.listdir(imagesPath):
  print(os.path.join(samplesPath, pathlib.Path(fn).stem+".mp4"))
  ffmpeg.input(os.path.join(imagesPath, fn),loop=1).output(os.path.join(samplesPath, pathlib.Path(fn).stem+".mp4"),vcodec="libx264",t=1,pix_fmt="yuv420p",vf=f'crop=iw:iw*9/16,scale={crop_width}:{crop_height}').run()