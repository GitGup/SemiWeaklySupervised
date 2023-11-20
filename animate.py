from common import *

def create_gif_fitout(sigfrac, m1, m2, w1, w2, z, epochs):
    output_directory = '2dhist_images'
    os.makedirs(output_directory, exist_ok=True)

    for key, value in train_wsmodel(sigfrac, m1, m2, w1, w2, epochs).items():
        w1_fit_coord = value[0]
        w2_fit_coord = value[1]

        loss_landscape_2D(sigfrac, m1, m2, w1_fit_coord, w2_fit_coord, z)

        image_path = os.path.join(output_directory, f'hist_{key}.png')
        plt.savefig(image_path)
        plt.close()

        clear_output(wait=True)
        
    #make the gif
    frames = []

    image_dir = '2dhist_images'
    image_files = os.listdir(image_dir)

    image_files = sorted([filename for filename in os.listdir(image_dir) if filename.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in image_files:
        image = Image.open(os.path.join(image_dir, filename))
        frames.append(image)
        os.remove(image_dir+"/"+filename)

    output_gif_filename = f'{sigfrac, m1, m2, w1, w2}.gif'
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=200, loop=0)

#animate loss landscape over different signal fractions
def create_gif_nofit(sigfrac, m1, m2, z):
    output_directory = '2dhist_images'
    os.makedirs(output_directory, exist_ok=True)
    
    frames = []  # List to store frames for all sigfrac values

    loss_landscape_2D_nofit(sigfrac, m1, m2, z)

    image_path = os.path.join(output_directory, f'hist_{sigfrac}.png')
    plt.savefig(image_path)
    plt.close()
    clear_output(wait=True)

    # Append the image to the frames list
    frames.append(Image.open(image_path))

    return frames