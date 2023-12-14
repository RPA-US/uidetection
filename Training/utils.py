import json
import os
import shutil

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, Point, Polygon


def create_slices(
    dataset_origin_path, dataset_dest_path, rows, columns, width_overlap, height_overlap, dataset_type
):
    aux_dict = {}

    for file in os.listdir(dataset_origin_path):
        aux_dict[file.split(".")[0] + ".png"] = [file.split(".")[0] + ".json"]

    for image in aux_dict.keys():
        annotations = json.load(
            open(os.path.join(dataset_origin_path, aux_dict[image][0]))
        )

        img = cv2.imread(os.path.join(dataset_origin_path, image))

        slice_width = int(img.shape[1] / columns)
        slice_height = int(img.shape[0] / rows)

        for i in range(rows):
            for j in range(columns):
                xmin = int(j * slice_width - width_overlap * slice_width)
                ymin = int(i * slice_height - height_overlap * slice_height)
                xmax = int((j + 1) * slice_width + width_overlap * slice_width)
                ymax = int((i + 1) * slice_height + height_overlap * slice_height)

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > img.shape[1]:
                    xmax = img.shape[1]
                if ymax > img.shape[0]:
                    ymax = img.shape[0]

                slice_img = img[ymin:ymax, xmin:xmax]
                slice_name = image.split(".")[0] + "_" + str(i) + "_" + str(j) + ".png"

                slice_annotations = {}
                slice_annotations["version"] = annotations["version"]
                slice_annotations["flags"] = annotations["flags"]
                slice_annotations["imageData"] = annotations["imageData"]
                slice_annotations["imagePath"] = slice_name
                slice_annotations["imageWidth"] = slice_img.shape[1]
                slice_annotations["imageHeight"] = slice_img.shape[0]
                slice_annotations["text"] = ""
                slice_annotations["shapes"] = []

                slice_poly = Polygon(
                    [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                )

                for shape in annotations["shapes"]:
                    intersections = slice_poly.intersection(
                        Polygon(shape["points"])
                    )

                    # We might get a single polygon or a collection of polygons, so we nomalize it
                    if type(intersections) != GeometryCollection:
                        intersections = GeometryCollection([intersections])
                    if any(intersection.area > 0 for intersection in intersections.geoms):

                        for intersection in intersections.geoms:
                            # We might get LineStrings or Points, so we ignore them
                            if type(intersection) != Polygon:
                                continue

                            intersection = intersection.buffer(0)

                            points = [
                                list(p) for p in list(intersection.exterior.coords)
                            ]
                            for point in points:
                                point[0] = point[0] - xmin
                                point[1] = point[1] - ymin

                                if point[0] < 0:
                                    point[0] = 0
                                elif point[0] > xmax - xmin:
                                    point[0] = xmax - xmin

                                if point[1] < 0:
                                    point[1] = 0
                                elif point[1] > ymax - ymin:
                                    point[1] = ymax - ymin

                            slice_shape = {
                                "label": shape["label"],
                                "text": shape["text"],
                                "points": points,
                                "group_id": None,
                                "shape_type": "polygon",
                                "flags": {},
                            }
                        if dataset_type == "seg":
                            slice_annotations["shapes"].append(slice_shape)
                        elif dataset_type == "bbox":
                            slice_shape["points"] = slice_shape["points"][:4]
                            slice_annotations["shapes"].append(slice_shape)
                        else:
                            raise ValueError("Dataset type not supported")

                # save only slices with annotations
                if len(slice_annotations["shapes"]) > 0:
                    cv2.imwrite(os.path.join(dataset_dest_path, slice_name), slice_img)
                    json.dump(
                        slice_annotations,
                        open(
                            os.path.join(
                                dataset_dest_path, slice_name.split(".")[0] + ".json"
                            ),
                            "w",
                        ),
                    )

                # Save original image and a annotations
                cv2.imwrite(os.path.join(dataset_dest_path, image), img)
                json.dump(
                    annotations,
                    open(os.path.join(dataset_dest_path, aux_dict[image][0]), "w"),
                )


def resize_dataset_images(path_to_dataset, path_to_preprocessed_dataset, width, height):
    """
    Resize all images and annotations in the dataset to a specific width and height
    :param path_to_dataset: Path to the dataset
    :param width: Width to resize the images to
    :param height: Height to resize the images to
    :return: None
    """

    # Loop over all the images
    for file in os.listdir(path_to_dataset):
        if file.endswith(".png"):
            # Open the image
            img = cv2.imread(path_to_dataset + "/" + file)

            # Resize the image with black zero-padding
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            # Save the image in the new dataset
            cv2.imwrite(path_to_preprocessed_dataset + "/" + file, img)

        elif file.endswith(".json"):
            # Loop over all the annotations
            labels = json.load(open(path_to_dataset + "/" + file))

            # Loop over all the bounding boxes
            for shape in labels["shapes"]:
                for point in shape["points"]:
                    # Update the point to match new dimensions
                    point[0] = point[0] * width / labels["imageWidth"]
                    point[1] = point[1] * height / labels["imageHeight"]

            # Update the size
            labels["imageWidth"] = width
            labels["imageHeight"] = height

            # Save the annotation in the new dataset
            with open(path_to_preprocessed_dataset + "/" + file, "w") as outfile:
                json.dump(labels, outfile)


def hue_augmentation(
    dataset_origin_path, dataset_destination_path, precentage, max_offset
):
    """
    Augment the dataset with hue augmentation
    :param dataset_origin_path: Path to the original dataset
    :param dataset_destination_path: Path to save the augmented dataset
    :param precentage: Precentage of images to augment
    :param max_offset: Maximum offset of hue
    :return: None
    """

    # Loop over all the images
    for file in os.listdir(dataset_origin_path):
        if file.endswith(".png"):
            # Open the image
            img = cv2.imread(dataset_origin_path + "/" + file)

            # Convert the image to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Generate a random number
            random_number = np.random.randint(0, 100)

            # Augment the image
            if random_number <= precentage * 100:
                # Select a random offset
                offset = np.random.randint(-max_offset, max_offset)

                # Add the offset to the hue channel
                img[:, :, 0] = img[:, :, 0] + offset

                # Convert the image back to BGR
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

                # Save the image in the new dataset
                cv2.imwrite(
                    dataset_destination_path
                    + "/"
                    + file.split(".")[0]
                    + "_hue_augmented.png",
                    img,
                )

                # Copy the json file to new dataset with the imagePath changed
                ann = json.load(
                    open(dataset_origin_path + "/" + file.split(".")[0] + ".json")
                )
                ann["imagePath"] = file.split(".")[0] + "_hue_augmented.png"
                with open(
                    dataset_destination_path
                    + "/"
                    + file.split(".")[0]
                    + "_hue_augmented.json",
                    "w",
                ) as outfile:
                    json.dump(ann, outfile)


def contrast_inversion_augmentation(
    dataset_origin_path, dataset_destination_path, precentage
):
    """
    Augment the dataset with contrast inversion augmentation
    :param dataset_origin_path: Path to the original dataset
    :param dataset_destination_path: Path to save the augmented dataset
    :param precentage: Precentage of images to augment
    :return: None
    """

    # Loop over all the images
    for file in os.listdir(dataset_origin_path):
        if file.endswith(".png"):
            # Open the image
            img = cv2.imread(dataset_origin_path + "/" + file)

            # Generate a random number
            random_number = np.random.randint(0, 100)

            # Augment the image
            if random_number <= precentage * 100:
                # Invert the contrast
                img = 255 - img

                # Save the image in the new dataset
                cv2.imwrite(
                    dataset_destination_path
                    + "/"
                    + file.split(".")[0]
                    + "_contrast_inversion_augmented.png",
                    img,
                )

                # Copy the json file to new dataset with the imagePath changed
                ann = json.load(
                    open(dataset_origin_path + "/" + file.split(".")[0] + ".json")
                )
                ann["imagePath"] = (
                    file.split(".")[0] + "_contrast_inversion_augmented.png"
                )
                with open(
                    dataset_destination_path
                    + "/"
                    + file.split(".")[0]
                    + "_contrast_inversion_augmented.json",
                    "w",
                ) as outfile:
                    json.dump(ann, outfile)


def labelme_to_yolo(
    labelme_dataset_path, yolo_dataset_path, train_split, classes, dataset_type
):
    """
    Convert a labelme dataset to a yolo dataset
    :param labelme_dataset_path: Path to the labelme dataset
    :param yolo_dataset_path: Path to save the yolo dataset
    :param train_split: Precentage of images to use for training
    :param classes: List of classes
    :return: None
    """

    # Create the yolo dataset directory
    if not os.path.exists(yolo_dataset_path):
        os.mkdir(yolo_dataset_path)

    # Create the yolo dataset train directory
    if not os.path.exists(yolo_dataset_path + "/train"):
        os.mkdir(yolo_dataset_path + "/train")
        os.mkdir(yolo_dataset_path + "/train/images")
        os.mkdir(yolo_dataset_path + "/train/labels")

    # Create the yolo dataset validation directory
    if not os.path.exists(yolo_dataset_path + "/val"):
        os.mkdir(yolo_dataset_path + "/val")
        os.mkdir(yolo_dataset_path + "/val/images")
        os.mkdir(yolo_dataset_path + "/val/labels")

    # Create the data.yml file
    with open(yolo_dataset_path + "/data.yaml", "w") as f:
        f.write("path: ../" + yolo_dataset_path + "\n")
        f.write("train: " + "train/images" + "\n")
        f.write("val: " + "val/images" + "\n")
        f.write("names:" + "\n")
        for i, v in enumerate(classes):
            f.write(f"   {i}: {v}" + "\n")

    # Loop over all the images
    for file in os.listdir(labelme_dataset_path):
        if file.endswith(".png"):
            # Open the image
            img = cv2.imread(labelme_dataset_path + "/" + file)

            # Decide if the image is for training or validation
            img_set = ""
            random_number = np.random.randint(0, 100)
            if random_number <= train_split * 100:
                cv2.imwrite(
                    yolo_dataset_path + "/train/images/" + file.split(".")[0] + ".jpg",
                    img,
                )
                open(
                    yolo_dataset_path + "/train/labels/" + file.split(".")[0] + ".txt",
                    "w+",
                ).close()
                img_set = "train"
            else:
                cv2.imwrite(
                    yolo_dataset_path + "/val/images/" + file.split(".")[0] + ".jpg",
                    img,
                )
                open(
                    yolo_dataset_path + "/val/labels/" + file.split(".")[0] + ".txt",
                    "w+",
                ).close()
                img_set = "val"

            # Loop over all the shapes
            ann = json.load(
                open(labelme_dataset_path + "/" + file.split(".")[0] + ".json")
            )

            if dataset_type == "seg":
                save_yolo_dataset_seg(
                    yolo_dataset_path, classes, ann, img, img_set, file
                )
            elif dataset_type == "bbox":
                save_yolo_dataset_bbox(
                    yolo_dataset_path, classes, ann, img, img_set, file
                )
            else:
                raise ValueError("Dataset type not supported")


def save_yolo_dataset_bbox(yolo_dataset_path, classes, ann, img, img_set, file):
    for shape in ann["shapes"]:
        if len(shape["points"]) != 4:
            print(
                "Skipping "
                + file
                + " because it has a shape with "
                + str(len(shape["points"]))
                + " points"
            )
        # Get the class index
        class_index = str(classes.index(shape["label"]))

        # Transform the points to yolo bbox format (x_center, y_center, width, height)
        x_center = str(
            round(
                ((shape["points"][0][0] + shape["points"][1][0]) / 2) / img.shape[1], 6
            )
        )
        y_center = str(
            round(
                ((shape["points"][0][1] + shape["points"][2][1]) / 2) / img.shape[0], 6
            )
        )
        width = str(
            round(abs(shape["points"][0][0] - shape["points"][1][0]) / img.shape[1], 6)
        )
        height = str(
            round(abs(shape["points"][0][1] - shape["points"][2][1]) / img.shape[0], 6)
        )

        ann_str = " ".join([class_index, x_center, y_center, width, height])

        # Save the annotation
        with open(
            yolo_dataset_path
            + "/"
            + img_set
            + "/labels/"
            + file.split(".")[0]
            + ".txt",
            "a",
        ) as f:
            f.write(ann_str + "\n")


def save_yolo_dataset_seg(yolo_dataset_path, classes, ann, img, img_set, file):
    for shape in ann["shapes"]:
        # Get the class index
        class_index = str(classes.index(shape["label"]))

        # Transform the points to yolo format. They have to be between 0 and 1
        retval = [class_index]
        for i in shape["points"]:
            i[0] = round(float(i[0]) / img.shape[1], 6)
            i[1] = round(float(i[1]) / img.shape[0], 6)
            # i[0] and i[1] must be between 0 and 1
            if i[0] < 0:
                i[0] = 0.000000
            elif i[0] > 1:
                i[0] = 1.000000
            if i[1] < 0:
                i[1] = 0.000000
            elif i[1] > 1:
                i[1] = 1.000000

            i[0] = str(i[0])
            i[1] = str(i[1])
            retval.extend(i)

        ann_str = " ".join(retval)

        # Save the annotation
        with open(
            yolo_dataset_path
            + "/"
            + img_set
            + "/labels/"
            + file.split(".")[0]
            + ".txt",
            "a",
        ) as f:
            f.write(ann_str + "\n")