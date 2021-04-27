import random

import torch


class ReplayBuffer:
    """An image buffer that stores previously generated images."""

    def __init__(self, max_size: int=50) -> None:
        assert (max_size > 0)
        self.max_size = max_size
        self.buffer = []
    
    def __call__(self, images):
        # If the buffer size is 0, do nothing
        if self.max_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # If the buffer is not full,
            # keep inserting current images to the buffer
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # The buffer will return a previously stored image,
                # and insert the current image into the buffer
                if p > 0.5:  
                    random_id = random.randint(0, self.max_size - 1)
                    tmp = self.buffer[random_id].clone()
                    self.buffer[random_id] = image
                    return_images.append(tmp)
                # The buffer will return the current image
                else:
                    return_images.append(image)
        # Collect all the images and return
        return_images = torch.cat(return_images, 0)
        
        return return_images
