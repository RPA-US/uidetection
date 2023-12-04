from cProfile import label
from charset_normalizer import detect
import cv2
import numpy as np
from tqdm import tqdm
import utils
from shapely.geometry import Polygon


def detect_duplicates(detected_shapes):
    # Create a 2d numpy array of shape (len(detected_shapes), len(detected_shapes))
    duplicates = []
    for i, shape in enumerate(detected_shapes):
        for shape2 in detected_shapes[i:]:
            detected_polygon1 = Polygon(shape["points"])
            detected_polygon2 = Polygon(shape2["points"])
            intersection = detected_polygon1.intersection(detected_polygon2).area
            union = detected_polygon1.union(detected_polygon2).area
            if union > 0 and intersection / union >= 0.5:
                duplicates.append((detected_polygon1, detected_polygon2))

    return duplicates


def get_orphaned_detections(mapping_matrix, detected_shapes):
    # Create a 1d numpy array of shape (len(detected_shapes),)
    orphaned_detections = []
    for i in range(mapping_matrix.shape[0]):
        if np.sum(mapping_matrix[i, :]) == 0:
            orphaned_detections.append(detected_shapes[i])

    return orphaned_detections


def get_orphaned_labels(mapping_matrix, labeled_shapes):
    # Create a 1d numpy array of shape (len(labeled_shapes),)
    orphaned_labels = []
    for i in range(mapping_matrix.shape[1]):
        if np.sum(mapping_matrix[:, i]) == 0:
            orphaned_labels.append(labeled_shapes[i])

    return orphaned_labels


def calculate_similarity_matrix(detected_shapes, labeled_shapes):
    # Create a 2d numpy array of shape (len(detected_shapes), len(labeled_shapes))
    similarity_matrix = np.zeros((len(detected_shapes), len(labeled_shapes)))
    for shape in detected_shapes:
        for shape2 in labeled_shapes:
            detected_polygon = Polygon(shape["points"])
            labeled_polygon = Polygon(shape2["points"])
            if detected_polygon.is_valid == False or labeled_polygon.is_valid == False:
                continue
            intersection = detected_polygon.intersection(labeled_polygon).area
            union = detected_polygon.union(labeled_polygon).area
            similarity_matrix[shape["id"], shape2["id"]] = intersection / union

    return similarity_matrix


def calculate_mapping_matrix(sm):
    mapping_matrix = np.zeros((sm.shape[0], sm.shape[1]))
    # Find the maximum value in each column and set it to 1. Remember each column and row correspond to a shape id
    for i in range(sm.shape[0]):
        max_index = np.argmax(sm[i, :])
        mapping_matrix[i, max_index] = 1 if sm[i, max_index] >= 0.5 else 0

    # For each column, if there is more than one keep only the one with the highest value
    for i in range(mapping_matrix.shape[1]):
        if np.sum(mapping_matrix[:, i]) > 1:
            max_index = np.argmax(sm[:, i])
            mapping_matrix[:, i] = 0
            mapping_matrix[max_index, i] = 1

    return mapping_matrix


def show_duplicates(img_path, duplicates):
    img = cv2.imread(img_path)
    for i in range(len(duplicates)):
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        cv2.polylines(img, np.int32([duplicates[i][0].exterior.coords]), True, color, 2)
        cv2.polylines(img, np.int32([duplicates[i][1].exterior.coords]), True, color, 2)

    cv2.imshow("duplicates", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_orpahns(img_path, orphan_labels, orphan_detections):
    tint_colors = {"labels": (0, 0, 255), "detections": (255, 0, 0)}
    img = cv2.imread(img_path)
    for i in range(len(orphan_labels)):
        cv2.polylines(
            img, np.int32([orphan_labels[i]["points"]]), True, tint_colors["labels"], 2
        )
    for i in range(len(orphan_detections)):
        cv2.polylines(
            img,
            np.int32([orphan_detections[i]["points"]]),
            True,
            tint_colors["detections"],
            2,
        )

    # Add a legend with the colors of the classes
    img_aux = img.copy()
    for i, label in enumerate(tint_colors):
        cv2.putText(
            img_aux,
            label,
            (10, 20 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            tint_colors[label],
            2,
        )

    img = cv2.addWeighted(img_aux, 0.6, img, 0.4, 0)

    cv2.imshow("orphans", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_mappings(img_path, detected_shapes, labeled_shapes, mapping_matrix):
    tint_colors = {}
    for i in range(len(detected_shapes)):
        if detected_shapes[i]["label"] not in tint_colors:
            tint_colors[detected_shapes[i]["label"]] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )

    img = cv2.imread(img_path)
    for i in range(mapping_matrix.shape[0]):
        for j in range(mapping_matrix.shape[1]):
            if mapping_matrix[i, j] == 1:
                # Show both polygons (labeled and detected)
                cv2.polylines(
                    img,
                    np.int32([detected_shapes[i]["points"]]),
                    True,
                    (255, 0, 0),
                    2,
                )
                cv2.polylines(
                    img,
                    np.int32([labeled_shapes[j]["points"]]),
                    True,
                    (0, 0, 255),
                    2,
                )

                # Tint the image with a color depending on its class to show the intersection
                detected_polygon = Polygon(detected_shapes[i]["points"])
                labeled_polygon = Polygon(labeled_shapes[j]["points"])
                intersection = detected_polygon.intersection(labeled_polygon)
                intersection_points = np.array(intersection.exterior.coords)

                img_aux = img.copy()
                cv2.fillPoly(
                    img_aux,
                    np.int32([intersection_points]),
                    tint_colors[detected_shapes[i]["label"]],
                )

                img = cv2.addWeighted(img_aux, 0.3, img, 0.7, 0)

    cv2.imshow("mappings", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_pth = "Detection metrics testing/data/captura.png"
    json_prefix = "Detection metrics testing/data/annotated_notext"

    (detected_json, labeled_json) = utils.load_json(json_prefix)
    similarity_matrix = calculate_similarity_matrix(
        detected_json["shapes"], labeled_json["shapes"]
    )
    mapping_matrix = calculate_mapping_matrix(similarity_matrix)
    duplicates = detect_duplicates(detected_json["shapes"])
    orphan_detection = get_orphaned_detections(mapping_matrix, detected_json["shapes"])
    orphan_labels = get_orphaned_labels(mapping_matrix, labeled_json["shapes"])

    show_duplicates(image_pth, duplicates)
    show_orpahns(image_pth, orphan_labels, orphan_detection)
    show_mappings(
        image_pth, detected_json["shapes"], labeled_json["shapes"], mapping_matrix
    )


def get_all_mapping_data(predicted_soms, dataset_soms):
    res = dict()
    for img_name in tqdm(dataset_soms.keys()):
        detected_json = predicted_soms[img_name]
        labeled_json = dataset_soms[img_name]
        similarity_matrix = calculate_similarity_matrix(
            detected_json["shapes"], labeled_json["shapes"]
        )
        mapping_matrix = calculate_mapping_matrix(similarity_matrix)
        duplicates = detect_duplicates(detected_json["shapes"])
        orphan_detection = get_orphaned_detections(mapping_matrix, detected_json["shapes"])
        orphan_labels = get_orphaned_labels(mapping_matrix, labeled_json["shapes"])

        res[img_name] = {
            "similarity_matrix": similarity_matrix,
            "mapping_matrix": mapping_matrix,
            "duplicates": duplicates,
            "orphan_detection": orphan_detection,
            "orphan_labels": orphan_labels,
        }
    
    return res
