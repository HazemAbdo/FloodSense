# TODOs
- [ ] split 80/10/10 vs 60/20/20
- [ ] add more data augmentation
    - The labeled dataset was heavily augmented to get more images for training the model under supervision.
    - The image samples were randomly cropped, shifted, resized and flipped along the horizontal and vertical axes.
- [ ] weighted sampling strategy (Both the classes were sampled equally during batch generation.)
- [ ] We downsized the image from 3000×4000 to 300×400 dimensions
- [ ] ResNet18 vs ResNet50
- [ ] A modified form of Binary Cross-Entropy (BCE) ??!!!
- [ ] use Model class 

- [ ] To expedite the process of feature extraction for the deep learning model, we apply a bilateral filter to the image, followed by two iterations of dilation and one iteration of erosion.
- [ ] For image augmentation we perform shuffling, rotation, scaling, shifting and brightness contrast.
- [ ] The images and masks are resized to 512×512 dimensions while training since high-resolution images preserve useful information.
- 