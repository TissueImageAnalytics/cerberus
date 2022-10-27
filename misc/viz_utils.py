import cv2
import math
import random
import colorsys
import numpy as np
import itertools
import yaml
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interp

from sklearn.metrics import roc_curve, auc, RocCurveDisplay

from .utils import get_bounding_box, center_pad_to_shape


__colour_dict = {
    "Nuclei": {
        "line_thickness": 2,
        "inst_colour": (255, 255, 0),
        "type_colour": {
            0: (0, 0, 0),
            1: (187, 17, 220),
            2: (0, 255, 0),
            3: (255, 0, 0),
            4: (0, 255, 255),
            5: (0, 0, 255),
            6: (255, 165, 0),
        },
    },
    "Gland": {
        "line_thickness": 3,
        "inst_colour": (0, 255, 0),
        "type_colour": {0: (0, 0, 0), 1: (255, 255, 0), 2: (255, 51, 153),},
    },
    "Lumen": {
        "line_thickness": 3,
        "inst_colour": (255, 0, 255),
        "type_colour": {0: (0, 0, 0),},
    },
}


def class_colour(class_value):
    """
    Generate RGB colour for overlay based on class id
    Args:
        class_value: integer denoting the class of object
    """
    if class_value == 0:
        return 0, 0, 0  # black (background)
    if class_value == 1:
        return 0, 165, 255  # orange
    elif class_value == 2:
        return 0, 255, 0  # green
    elif class_value == 3:
        return 0, 0, 255  # blue
    elif class_value == 4:
        return 255, 255, 0  # yellow
    elif class_value == 5:
        return 255, 0, 0  # red
    elif class_value == 6:
        return 0, 255, 255  # cyan
    else:
        raise Exception(
            "Currently, overlay_segmentation_results() only supports up to 6 classes."
        )


def random_colors(N, bright=True):
    """Generate random colors.
    
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def colorize(ch, vmin, vmax, cmap=plt.get_cmap("jet"), shape=None):
    """
    Will clamp value value outside the provided range to vmax and vmin
    """
    ch = np.squeeze(ch.astype("float32"))
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    if shape is not None:
        ch_cmap = center_pad_to_shape(ch_cmap, shape)
    return ch_cmap


def visualize_instances_map(
    input_image, inst_map, type_map=None, type_colour=None, line_width=2
):
    """Overlays segmentation results on image as contours.
    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_width: line width of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    if 0 in inst_list:
        inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        if contours_crop.size == 2:
            contours_crop = np.expand_dims(contours_crop, 0)
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (255, 255, 0)
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_width)
    return overlay


def visualize_instances_dict(
    img, results_dict, src_tl, ds_factor=1, type_colour=None, line_width=2
):
    gland_colour = (255, 255, 0)
    lumen_colour = (0, 255, 0)

    overlay = np.copy((img).astype(np.uint8))
    # downscale image - do not need to save the overlay at a high resolution
    if ds_factor > 1:
        overlay = cv2.resize(img, (0, 0), fx=1 / ds_factor, fy=1 / ds_factor)

    for idx, [inst_id, inst_info] in enumerate(results_dict.items()):
        inst_gland_cnt = inst_info["gland_cnt"]
        # contours are wrt the wsi - convert so that are wrt the tile
        inst_gland_cnt -= np.array(src_tl)
        # if downscaling the overlay, coordinates need to be downscaled too
        if ds_factor > 1:
            inst_gland_cnt = inst_gland_cnt / ds_factor
        inst_gland_cnt = inst_gland_cnt.astype("int")
        cv2.drawContours(overlay, [inst_gland_cnt], -1, gland_colour, line_width)

        inst_lumen_cnt_list = inst_info["lumen_cnt"]
        nr_lumen = len(inst_lumen_cnt_list)
        if nr_lumen > 0:
            for inst_lumen_cnt in inst_lumen_cnt_list:
                # contours are wrt the wsi - convert so that are wrt the tile
                inst_lumen_cnt -= np.array(src_tl)
                # if downscaling the overlay, coordinates need to be downscaled too
                if ds_factor > 1:
                    inst_lumen_cnt = inst_lumen_cnt / ds_factor
                inst_lumen_cnt = inst_lumen_cnt.astype("int")
                cv2.drawContours(
                    overlay, [inst_lumen_cnt], -1, lumen_colour, line_width
                )
    return overlay


def visualize_instances_dict_orig(input_image, inst_dict_):
    overlay = np.copy((input_image).astype(np.uint8))


    with open("dataset.yml") as fptr:
        dataset_info = yaml.full_load(fptr)

    # enforce ordering of overlay
    for tissue in ["Gland", "Lumen", "Nuclei"]:
        if tissue in inst_dict_.keys():
            inst_dict = inst_dict_[tissue]
            for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
                inst_contour = inst_info["contour"]
                viz_info = dataset_info[tissue.lower()]['viz_info']
                line_width = viz_info["line_width"]
                if "type" in inst_info:
                    inst_colour = viz_info["type_colour"][
                        inst_info["type"]
                    ]
                else:
                    inst_colour = viz_info["inst_colour"]
                # only consider rgb
                inst_colour = inst_colour[:3]
                cv2.drawContours(
                    overlay, [inst_contour], -1, inst_colour, line_width
                )

    return overlay


def visualize_graph(
    vertices, edges, canvas=None, edge_color=(0, 255, 0), node_color=(255, 0, 0)
):
    """
    # TODO: error checking for xy in vertices against canvas size
    Args:
        vertices (np.array): Nx2 with N is the number of vertices, index 0 is x
                             and index 1 is y in Euclidean coordinate
        edges    (np.array): Nx2 with N is the number of edges, index 0 is index
                             of the source node, index 1 is the index of the target
                             node
        canvas   (np.array): image to plot the graph on
    """
    if canvas is None:
        x_vert = np.max(vertices[:, 0])
        y_vert = np.max(vertices[:, 1])
        canvas = np.zeros([int(round(y_vert)), int(round(x_vert)), 3])

    rounded_vertices = (vertices + 0.5).astype("int32")
    for edge in edges:
        cv2.line(
            canvas,
            tuple(rounded_vertices[edge[0]]),
            tuple(rounded_vertices[edge[1]]),
            edge_color,
            2,
        )
    for vertex in list(rounded_vertices):
        cv2.circle(canvas, tuple(vertex), 8, node_color, -1)
    return canvas


def gen_figure(
    imgs_list,
    titles,
    fig_inch,
    shape=None,
    share_ax="all",
    show=False,
    colormap=plt.get_cmap("jet"),
):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig


def get_vis(y, y_pred, mean_fp, name, ax, alpha=0.3, lw=1):
    fp, tp, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fp, tp)
    viz = RocCurveDisplay(fpr=fp, tpr=tp, roc_auc=roc_auc, estimator_name=name)

    interp_tp = interp(mean_fp, fp, tp)
    interp_tp[0] = 0.0

    return fp, interp_tp, roc_auc


def plot_roc(tp_list, auc_list, mean_fp, ax, save_path, title):
    fig, ax = plt.subplots()

    mean_tp = np.mean(tp_list, axis=0)
    mean_tp[-1] = 1.0
    mean_auc = auc(mean_fp, mean_tp)
    std_auc = np.std(auc_list)
    ax.plot(
        mean_fp,
        mean_tp,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tp = np.std(tp_list, axis=0)
    tp_upper = np.minimum(mean_tp + std_tp, 1)
    tp_lower = np.maximum(mean_tp - std_tp, 0)
    ax.fill_between(
        mean_fp,
        tp_lower,
        tp_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title,
    )
    ax.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_path)

