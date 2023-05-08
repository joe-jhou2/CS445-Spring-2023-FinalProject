Project Title: Image-to-Image and Video-to-Video Translation using Pix2Pix and CycleGAN

Team Members: Santiago Valencia Sanchez, Jue Hou, Nikhil Khandekar, Azaan Barlas

Final Results: https://www.youtube.com/shorts/hspKupwXvsM and https://www.youtube.com/shorts/hspKupwXvsM

Overview:
Our project explores the power of image-to-image and video-to-video translation using Pix2Pix and CycleGAN. We focused on three main areas of application: satellite-to-map images, facade segmentation, and artistic style transfer on videos.

Methodology:
We re-implemented Pix2Pix and CycleGAN and used non-ML image processing methods to help transform and stitch large images and perform video-to-video translation. 
For the map section, we input large satellite map images and broke them up into smaller patches with overlap (512x512pix with 50 overlapping). After processing these patches through the model, we used a stitching method to combine them and blend the overlap using gaussian-laplacian pyramids. 
For videos, we broke them up into individual frames, resized them to 512x512, and input them into the models. We utilized various techniques to improve the video quality, including histogram matching, gaussian-laplacian pyramids blending, and denoising by appling non-local means filter to remove noise.

Results:
Our results showed that CycleGAN outperformed Pix2Pix in both the map and artistic video sections. For the map section, both models generated maps that closely resembled Google maps. However, for artistic style transfer, CycleGAN performed better than Pix2Pix, maintaining the structures of objects in the pictures while having less noise. Pix2Pix required additional processing techniques to improve the video quality, while CycleGAN did not.

Conclusion:
Our project highlights the effectiveness of image-to-image and video-to-video translation using Pix2Pix and CycleGAN. By incorporating non-ML image processing methods, we were able to achieve better results with fewer resources. Our findings suggest that CycleGAN is a better choice for artistic style transfer and that Pix2Pix requires additional processing techniques for video output.
