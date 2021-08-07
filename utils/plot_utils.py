import numpy as np
import matplotlib.pyplot as plt

from config import Config


class PlotUtils:

    @staticmethod
    def plot_setup():
        room_config = Config.room
        doi_config = Config.doi
        ratio = room_config["length"] / doi_config["length"]
        grids = int(doi_config["forward_grids"] * ratio)

        image = np.ones((grids, grids))
        if doi_config["origin"] == "center":
            diff = int((grids - doi_config["forward_grids"])/2)
            image[diff:grids-diff, diff:grids-diff] = 0
        else:
            diff = int(grids/2)
            image[diff:doi_config["forward_grids"], diff:doi_config["forward_grids"]] = 0
        plt.imshow(image, cmap=plt.cm.gray, extent=PlotUtils.get_room_extent())
        plt.show()

    @staticmethod
    def get_doi_extent():
        doi_length = Config.doi["length"]
        doi_width = Config.doi["width"]
        if Config.doi["origin"] == "center":
            extent = [-doi_length/2, doi_length/2, -doi_width/2, doi_width/2]
        else:
            extent = [0, doi_length, 0, doi_width]
        return extent

    @staticmethod
    def get_room_extent():
        room_length = Config.room["length"]
        room_width = Config.room["width"]
        if Config.room["origin"] == "center":
            extent = [-room_length / 2, room_length / 2, -room_width / 2, room_width / 2]
        else:
            extent = [0, room_length, 0, room_width]
        return extent


if __name__ == '__main__':

    PlotUtils.plot_setup()
