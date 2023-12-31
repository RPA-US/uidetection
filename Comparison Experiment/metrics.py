import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from hierarchy_constructor import *
from mapping import *
from sklearn.metrics import ConfusionMatrixDisplay
from utils import *


def run_image(detections, directory, technique, output_dir, compare_classes=True):
    dataset_labels = load_dataset(directory)
    dataset_soms = labels_to_soms(copy.deepcopy(dataset_labels))

    # Get SOMs from detections
    predicted_soms = labels_to_soms(copy.deepcopy(detections))

    if not os.path.exists(output_dir + f"/{technique}/soms"):
        os.makedirs(output_dir + f"/{technique}/soms")
    for file, som in predicted_soms.items():
        if os.path.exists(output_dir + f"/{technique}/soms/som_{file}_detected.json"):
            os.remove(output_dir + f"/{technique}/soms/som_{file}_detected.json")
        json.dump(
            som, open(output_dir + f"/{technique}/soms/som_{file}_detected.json", "w")
        )

    mappings = get_all_mapping_data(detections, dataset_labels)

    # Get class labels
    labels = set()
    for img_name, img in dataset_labels.items():
        for shape in img["shapes"]:
            labels.add(shape["label"])
    labels = list(labels)

    save_class_metrics(
        detections,
        dataset_labels,
        mappings,
        labels,
        output_dir + f"/{technique}",
        compare_classes,
    )
    save_iou_metrics(
        detections,
        dataset_labels,
        mappings,
        labels,
        output_dir + f"/{technique}",
        compare_classes,
    )
    save_som_metrics(
        predicted_soms,
        dataset_soms,
        mappings,
        output_dir + f"/{technique}",
        compare_classes,
    )


def save_class_metrics(
    detections, dataset_labels, mappings, labels, output_dir, compare_classes
):
    # Claculate precision, recall and confusion matrix
    label_count = {label: 0 for label in labels}
    label_precision = {label: 0 for label in labels}
    label_recall = {label: 0 for label in labels}
    num_det_shapes = {label: 0 for label in labels}

    if compare_classes:
        # y_pred and y_true for confusion matrix
        y_pred = []
        y_true = []
    for img_name in dataset_labels.keys():
        shapes = dataset_labels[img_name]["shapes"]
        mapping_matrix = mappings[img_name]["mapping_matrix"]

        det_shapes = detections[img_name]["shapes"]
        for shape in det_shapes:
            if shape["label"] in labels:
                num_det_shapes[shape["label"]] += 1

        for shape in shapes:
            detected = np.sum(mapping_matrix[:, shape["id"]]) > 0
            label_count[shape["label"]] += 1

            if detected:
                detected_shape_id = np.argmax(mapping_matrix[:, shape["id"]])
                detected_shape = list(
                    filter(
                        lambda x: x["id"] == detected_shape_id,
                        detections[img_name]["shapes"],
                    )
                )[0]

                if compare_classes:
                    y_pred.append(detected_shape["label"])
                    y_true.append(shape["label"])

                if (not compare_classes) or shape["label"] == detected_shape["label"]:
                    label_precision[shape["label"]] += 1
                    label_recall[shape["label"]] += 1

    if compare_classes:
        label_precision = {
            label: label_precision[label] / num_det_shapes[label]
            if num_det_shapes[label] > 0
            else 0
            for label in label_precision
        }
        label_recall = {
            label: label_recall[label] / label_count[label] for label in label_recall
        }
        label_f1_score = {
            label: (2 * label_precision[label] * label_recall[label])
            / (label_precision[label] + label_recall[label] + 1e-10)
            for label in label_recall
        }
    else:
        detected_shapes = 0
        mapped_shapes = 0
        dataset_shapes = 0
        for img_name in dataset_labels.keys():
            detected_shapes += len(detections[img_name]["shapes"])
            mapped_shapes += np.sum(mappings[img_name]["mapping_matrix"] > 0)
            dataset_shapes += len(dataset_labels[img_name]["shapes"])

        label_precision = {
            "all": mapped_shapes / detected_shapes,
        }
        label_recall = {
            "all": mapped_shapes / dataset_shapes,
        }
        label_f1_score = {
            "all": (2 * label_precision["all"] * label_recall["all"])
            / (label_precision["all"] + label_recall["all"] + 1e-10),
        }

    # Save class count
    plt.figure(figsize=(10, 5))
    plt.bar(label_count.keys(), label_count.values())
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(output_dir + "/class_count.png", bbox_inches="tight")
    plt.close()

    # Save Precision
    plt.figure(figsize=(10, 5))
    plt.bar(label_precision.keys(), label_precision.values())
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.title("Precision per label")
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.savefig(output_dir + "/precision.png", bbox_inches="tight")
    plt.close()

    # Save Recall
    plt.figure(figsize=(10, 5))
    plt.bar(label_recall.keys(), label_recall.values(), color="#ff7f0e")
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.title("Recall per label")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.savefig(output_dir + "/recall.png", bbox_inches="tight")

    # Save F1 Score
    plt.figure(figsize=(10, 5))
    plt.bar(label_f1_score.keys(), label_f1_score.values(), color="#2ca02c")
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.title("F1 Score per label")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.savefig(output_dir + "/f1_score.png", bbox_inches="tight")
    plt.close()

    if compare_classes:
        # Save Confusion Matrix
        plt.rcParams["figure.figsize"] = [20, 10]
        cm = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize="true",
            xticks_rotation=90,
        )
        cm.figure_.savefig(output_dir + "/confusion_matrix.png", bbox_inches="tight")


def save_iou_metrics(
    detections, dataset_labels, mappings, labels, output_dir, compare_classes
):
    iou_acc = dict()

    for img_name in dataset_labels.keys():
        iou_acc[img_name] = dict()
        mappings_per_label = dict()
        for label in labels:
            iou_acc[img_name][label] = 0.0
            mappings_per_label[label] = 0

        # Get the mapped pairs from the mapping matrix
        mapping_matrix = mappings[img_name]["mapping_matrix"]
        mapped_pairs = np.argwhere(mapping_matrix > 0)

        for pair in mapped_pairs:
            detected_shape = list(
                filter(
                    lambda shape: shape["id"] == pair[0], detections[img_name]["shapes"]
                )
            )[0]

            dataset_shape = list(
                filter(
                    lambda shape: shape["id"] == pair[1],
                    dataset_labels[img_name]["shapes"],
                )
            )[0]

            if (not compare_classes) or detected_shape["label"] == dataset_shape[
                "label"
            ]:
                label = dataset_shape["label"]
                det_shape_polygon = Polygon(detected_shape["points"])
                dataset_shape_polygon = Polygon(dataset_shape["points"])
                mappings_per_label[label] += 1

                iou = (
                    det_shape_polygon.intersection(dataset_shape_polygon).area
                    / det_shape_polygon.union(dataset_shape_polygon).area
                )

                if np.isclose(iou_acc[img_name][label], 0.0, rtol=1e-09, atol=1e-09):
                    iou_acc[img_name][label] = iou
                else:
                    # Formula from https://math.stackexchange.com/questions/22348/how-to-add-and-subtract-values-from-an-average
                    iou_acc[img_name][label] = (
                        iou_acc[img_name][label]
                        + (iou - iou_acc[img_name][label]) / mappings_per_label[label]
                    )

    # Average IOU accuracy
    iou_acc_avg = dict()
    # We map the iuo accuracy dict to a list of values corresponding to the label, the filter by removing the 0 values, then we averagae
    values = list(map(lambda x: x[label], iou_acc.values()))
    if compare_classes:
        for label in labels:
            if len(values) == 0:
                iou_acc_avg[label] = 0.0
            else:
                iou_acc_avg[label] = np.average(list(filter(lambda x: x > 0, values)))
    else:
        iou_acc_avg["all"] = np.average(list(filter(lambda x: x > 0, values)))

    # Save IOU accuracy
    plt.figure(figsize=(10, 5))
    plt.bar(iou_acc_avg.keys(), iou_acc_avg.values())
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.title("IOU accuracy per label")
    plt.xlabel("Class")
    plt.ylabel("IOU accuracy")
    plt.savefig(output_dir + "/iou_accuracy.png", bbox_inches="tight")
    plt.close()


def get_tree_items(tree):
    items = []
    for item in tree:
        if item["type"] == "leaf":
            items.append(item)
        else:
            items.append(item)
            if len(item["children"]) > 0:
                items.extend(get_tree_items(item["children"]))
            else:
                item["type"] = "leaf"
    return items


def save_som_metrics(
    predicted_soms, dataset_soms, mappings, output_dir, compare_classes
):
    dataset_soms_items = dict()
    detected_soms_items = dict()

    for img_name in dataset_soms.keys():
        dataset_soms_items[img_name] = get_tree_items(
            dataset_soms[img_name]["children"]
        )
        detected_soms_items[img_name] = get_tree_items(
            predicted_soms[img_name]["children"]
        )

    som_detection_metrics = {
        "depth_acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "missed_children": 0.0,
        "detection_acc": 0.0,
        "false_det": {"total": 0, "class": 0.0, "segment": 0.0},
    }

    depth_acc = dict()
    recall = dict()
    precision = dict()
    missed_children = dict()
    detection_acc = dict()
    false_det = dict()

    for img_name in dataset_soms_items.keys():
        depth_acc[img_name] = 0.0
        # Recall and precision are calculated per node and then weighted averaged by the number of children
        recall[img_name] = dict()
        precision[img_name] = dict()

        # Weights for recall and precision
        weights = []

        missed_children[img_name] = 0.0
        detection_acc[img_name] = 0.0
        false_det[img_name] = {"total": 0, "class": 0.0, "segment": 0.0}

        mapping_matrix = mappings[img_name]["mapping_matrix"]
        orphan_detections = mappings[img_name]["orphan_detection"]
        duplicates = mappings[img_name]["duplicates"]

        non_duplicate_orphans = list(
            filter(lambda x: not any(x in d for d in duplicates), orphan_detections)
        )

        dataset_items = dataset_soms_items[img_name]
        detected_items = detected_soms_items[img_name]

        for shape in dataset_items:
            mapped_node_id = np.argmax(mapping_matrix[:, shape["id"]])
            mapped_node = list(
                filter(lambda n: n["id"] == mapped_node_id, detected_items)
            )[0]

            # Calculate recall and precision in a per-node basis
            if shape["type"] == "root" or shape["type"] == "node":
                recall[img_name][shape["id"]] = 0.0
                precision[img_name][shape["id"]] = 0.0

                for child in shape["children"]:
                    if np.sum(mapping_matrix[:, child["id"]]) > 0:
                        mapped_shape_id = np.argmax(mapping_matrix[:, child["id"]])
                        mapped_shape = list(
                            filter(lambda s: s["id"] == mapped_shape_id, detected_items)
                        )[0]

                        if mapped_shape in mapped_node["children"]:
                            if (not compare_classes) or mapped_shape["label"] == child[
                                "label"
                            ]:
                                recall[img_name][shape["id"]] += 1
                                precision[img_name][shape["id"]] += 1
                            else:
                                false_det[img_name]["class"] += 1

                        else:
                            if (not compare_classes) or mapped_shape["label"] == child[
                                "label"
                            ]:
                                missed_children[img_name] += 1

                # Normalize recall and precision and add weights
                recall[img_name][shape["id"]] /= len(shape["children"])
                if len(mapped_node["children"]) > 0:
                    precision[img_name][shape["id"]] /= len(mapped_node["children"])
                else:
                    precision[img_name][shape["id"]] = 0

                weights.append(len(shape["children"]))

            # Calculate the rest of metrics in the same way
            if np.sum(mapping_matrix[:, shape["id"]]) > 0:
                if (not compare_classes) or mapped_node["label"] == shape["label"]:
                    if mapped_node["depth"] == shape["depth"]:
                        depth_acc[img_name] += 1

                    detection_acc[img_name] += 1

                else:
                    false_det[img_name]["class"] += 1

        false_det[img_name]["segment"] = len(non_duplicate_orphans)
        false_det[img_name]["total"] += (
            false_det[img_name]["class"] + false_det[img_name]["segment"]
        )

        # Normalize values
        # Right now we can use detection_acc as the number of relevant retrieved instances
        depth_acc[img_name] /= len(dataset_items)
        detection_acc[img_name] /= len(dataset_items)

        false_det[img_name]["total"] /= len(detected_items)
        false_det[img_name]["class"] /= len(detected_items)
        false_det[img_name]["segment"] /= len(detected_items)

        # Weighted average for recall and precision
        recall[img_name] = np.average(list(recall[img_name].values()), weights=weights)
        precision[img_name] = np.average(
            list(precision[img_name].values()), weights=weights
        )

        missed_children[img_name] /= len(dataset_items)
    # Calculate averaged metrics
    som_detection_metrics["depth_acc"] = np.average(list(depth_acc.values()))
    som_detection_metrics["precision"] = np.average(list(precision.values()))
    som_detection_metrics["recall"] = np.average(list(recall.values()))
    som_detection_metrics["f1_score"] = (
        2 * som_detection_metrics["precision"] * som_detection_metrics["recall"]
    ) / (som_detection_metrics["precision"] + som_detection_metrics["recall"] + 1e-10)
    som_detection_metrics["missed_children"] = np.average(
        list(missed_children.values())
    )
    som_detection_metrics["detection_acc"] = np.average(list(detection_acc.values()))
    som_detection_metrics["false_det"]["total"] = np.average(
        [false_det[img_name]["total"] for img_name in false_det.keys()]
    )
    som_detection_metrics["false_det"]["class"] = np.average(
        [false_det[img_name]["class"] for img_name in false_det.keys()]
    )
    som_detection_metrics["false_det"]["segment"] = np.average(
        [false_det[img_name]["segment"] for img_name in false_det.keys()]
    )

    # General metrics
    som_detection_metrics_no_false_det = copy.deepcopy(som_detection_metrics)
    del som_detection_metrics_no_false_det["false_det"]

    # Save SOM metrics
    plt.figure(figsize=(10, 5))
    plt.bar(
        som_detection_metrics_no_false_det.keys(),
        som_detection_metrics_no_false_det.values(),
    )
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("SOM detection metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.savefig(output_dir + "/som_metrics.png", bbox_inches="tight")
    plt.close()

    # False detections
    plt.figure(figsize=(10, 5))
    plt.bar(
        som_detection_metrics["false_det"].keys(),
        som_detection_metrics["false_det"].values(),
    )
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("SOM false detections")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.savefig(output_dir + "/som_false_detections.png", bbox_inches="tight")
    plt.close()
