import getopt, sys
from tkinter import image_names

import screen2som
import yolo
import mobileSAM
import uied
import metrics
import os


def image_experiment(directory):
    output_dir = f"runs/{directory.split('.')[0]}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run screen2som
    screen2som_detections = screen2som.predict(directory, output_dir)
    metrics.run_image(screen2som_detections, directory, "screen2som", output_dir)
    metrics.run_image(
        screen2som_detections,
        directory,
        "screen2som-no-compare-classes",
        output_dir,
        compare_classes=False,
    )

    # Run YOLO detections
    yolo_detections = yolo.predict(directory, output_dir)
    metrics.run_image(
        yolo_detections, directory, "yolo", output_dir, compare_classes=False
    )

    # Run MobileSAM detections
    mobilesam_detections = mobileSAM.predict(directory, output_dir)
    metrics.run_image(
        mobilesam_detections, directory, "mobileSAM", output_dir, compare_classes=False
    )

    # Run UIED detections
    uied_detections = uied.batch_component_detection(directory, output_dir)
    metrics.run_image(
        uied_detections, directory, "UIED", output_dir, compare_classes=False
    )

    # TODO: Run rpa-us detections


if __name__ == "__main__":
    # Remove 1st argument from the
    # list of command line arguments
    argument_list = sys.argv[1:]

    # Options
    options = "hd:"

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
                image_experiment(current_value)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
