import getopt
import logging as log
import os
import sys
import time
from turtle import st

import kevin_moran
import metrics
import mobileSAM
import screen2som
import uied
import yolo
from utils import *


def image_experiment(directory):
    output_dir = f"runs/{directory.split('/')[-1]}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create csv to store precision, recall, f1 score and iou accuracy per technique
    with open(output_dir + "metrics.csv", "w") as f:
        f.write("Technique, Time, Avg Precision, Avg Recall, Avg F1 Score, Avg IOU Accuracy, Avg Area Perc. Det., Avg SOM Depth Acc,  Avg SOM Press, Avg SOM Rec., Avg SOM F1, Avg SOM missed children, Avg SOM Det. Acc., Avg SOM False Pos., Avg SOM False Class Pos, Avg SOM False Seg. Pos., Avg Edit Tree Distance\n")

    # Run screen2som
    log.info("Running screen2som")
    start_time = time.time()
    screen2som_detections = screen2som.predict(directory, output_dir + "screen2som")
    end_time = time.time()
    execution_time = end_time - start_time
    # Save borders
    save_detections(output_dir + "screen2som/detections", directory, output_dir + "screen2som/borders")
    # Save in csv
    with open(output_dir + "metrics.csv", "a") as f:
        f.write(f"screen2som-no-compare-classes, {execution_time},")
    # metrics.run_image(screen2som_detections, directory, "screen2som", output_dir)
    metrics.run_image(
        screen2som_detections,
        directory,
        "screen2som-no-compare-classes",
        output_dir,
        compare_classes=False,
    )

    # Run YOLO detections
    log.info("Running YOLO")
    start_time = time.time()
    yolo_detections = yolo.predict(directory, output_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    # Save borders
    save_detections(output_dir + "yolo/detections", directory, output_dir + "yolo/borders")
    # Save in csv
    with open(output_dir + "metrics.csv", "a") as f:
        f.write(f"yolo, {execution_time},")
    metrics.run_image(
        yolo_detections, directory, "yolo", output_dir, compare_classes=False
    )

    # Run MobileSAM detections
    log.info("Running MobileSAM")
    start_time = time.time()
    mobilesam_detections = mobileSAM.predict(directory, output_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    # Save borders
    save_detections(output_dir + "mobileSAM/detections", directory, output_dir + "mobileSAM/borders")
    # Save in csv
    with open(output_dir + "metrics.csv", "a") as f:
        f.write(f"mobileSAM, {execution_time},")
    metrics.run_image(
        mobilesam_detections, directory, "mobileSAM", output_dir, compare_classes=False
    )

    # Run UIED detections
    log.info("Running UIED")
    start_time = time.time()
    uied_detections = uied.batch_component_detection(directory, output_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    # Save borders
    save_detections(output_dir + "UIED/detections", directory, output_dir + "UIED/borders")
    # Save in csv
    with open(output_dir + "metrics.csv", "a") as f:
        f.write(f"UIED, {execution_time},")
    metrics.run_image(
      uied_detections, directory, "UIED", output_dir, compare_classes=False
    )

    # Run kevin-moran detections
    log.info("Running kevin-moran")
    start_time = time.time()
    kevin_moran_detections = kevin_moran.predict(directory, output_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    # Save borders
    save_detections(output_dir + "kevin_moran/detections", directory, output_dir + "kevin_moran/borders")
    # Save in csv
    with open(output_dir + "metrics.csv", "a") as f:
        f.write(f"kevin-moran, {execution_time},")
    metrics.run_image(
        kevin_moran_detections, directory, "kevin_moran", output_dir, compare_classes=False
    )


if __name__ == "__main__":
    # Remove 1st argument from the
    # list of command line arguments
    argument_list = sys.argv[1:]

    # Options
    options = "hd:i:"

    # Long options
    long_options = ["help", "directory"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, options, long_options)

        # checking each argument
        for current_argument, current_value in arguments:
            if current_argument in ("-h", "--help"):
                print("Displaying Help")

            elif current_argument in ("-d", "--directory"):
                # If argument "-i" or "--iterative" is passed, loop over the images
                image_experiment(current_value)
            elif current_argument in ("-i", "--iterative"):
                for dir in os.listdir(current_value):
                    image_experiment(os.path.join(current_value, dir).replace("\\", "/"))

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
