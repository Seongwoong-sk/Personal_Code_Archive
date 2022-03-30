# DataFrame에서 list로 정보를 받아와서 python으로 그리기

import matplotlib.pyplot as plt

img_to_draw = [ input_path + '/' + file for file in train_df.query("individual_id == '281504409737'").sample(25).image]
print(type(img_to_draw))

fig, axes = plt.subplots(5, 5, figsize=(20,20))

for idx, img in enumerate(img_to_draw):
    i = idx % 5 
    j = idx // 5
    image = Img.open(img)
    iar_shp = np.array(image).shape
    axes[i, j].axis('off')
    axes[i, j].imshow(image)
    
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()