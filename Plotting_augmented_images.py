def plot_augmentation(image_id, transform):
    plt.figure(figsize=(16, 4))
    img = cv2.imread(os.path.join(BASE_DIR, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.show()
    

import albumentations as A
    
transform_shift_scale_rotate = A.ShiftScaleRotate(
    p=1.0, 
    shift_limit=(-0.3, 0.3), 
    scale_limit=(-0.1, 0.1), 
    rotate_limit=(-180, 180), 
    interpolation=0, 
    border_mode=4, 
)

plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)


transform_coarse_dropout = A.CoarseDropout(
    p=1.0, 
    max_holes=100, 
    max_height=50, 
    max_width=50, 
    min_holes=30, 
    min_height=20, 
    min_width=20,
)

plot_augmentation("1003442061.jpg", transform_coarse_dropout)


## Using Case ##
transform = A.Compose(
    transforms=[
        transform_shift_scale_rotate,
        transform_coarse_dropout,
    ],
    p=1.0,
)

plot_augmentation("1003442061.jpg", transform)
