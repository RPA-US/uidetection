import os
import json
import keras_ocr
import cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

def predict(directory, output_dir, cropping_threshold=2):
    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir + "/kevin_moran"):
        os.makedirs(output_dir + "/kevin_moran")
    if not os.path.exists(output_dir + "/kevin_moran/detections"):
        os.makedirs(output_dir + "/kevin_moran/detections")

    for img_name, img_path in tqdm(images.items(), desc="Running kevin-moran predictions"):
        ocr_result = get_ocr_image(img_path)
        texto_detectado_ocr = ocr_result[0]

        image_pil = cv2.imread(img_path)
        detections[img_name] = dict()
        detections[img_name]["imageWidth"] = int(image_pil.shape[1])
        detections[img_name]["imageHeight"] = int(image_pil.shape[0])
        detections[img_name]["shapes"] = get_gui_components_crops(img_path, img_name, texto_detectado_ocr, cropping_threshold)

        # Save detections
        json.dump(
            detections[img_name],
            open(output_dir + f"/kevin_moran/detections/detections_{img_name}.json", "w"),
        )

    return detections

def get_gui_components_crops(img_path, img_name, texto_detectado_ocr, cropping_threshold):
    '''
    Analyzes an image and extracts its UI components

    :param param_img_root: Path to the image
    :type param_img_root: str
    :param image_names: Names of the images in the path
    :type image_names: list
    :param texto_detectado_ocr: Text detected by OCR in previous step
    :type texto_detectado_ocr: list
    :param path_to_save_bordered_images: Path to save the image along with the components detected
    :type path_to_save_bordered_images: str
    :param img_name: Index of the image we want to analyze in images_names
    :type img_name: int
    :return: Crops and text inside components
    :rtype: Tuple
    '''
    words = {}

    # Read the image
    img = cv2.imread(img_path)

    # Store on global_y all the "y" coordinates and text boxes
    # Each row is a different text box, much more friendly than the format returned by keras_ocr 
    global_y = []
    global_x = []
    words[img_name] = {}

    for j in range(0, len(texto_detectado_ocr)):
        coordenada_y = []
        coordenada_x = []

        for i in range(0, len(texto_detectado_ocr[j][1])):
            coordenada_y.append(texto_detectado_ocr[j][1][i][1])
            coordenada_x.append(texto_detectado_ocr[j][1][i][0])

        word = texto_detectado_ocr[j][0]
        centroid = (np.mean(coordenada_x), np.mean(coordenada_y))
        if word in words[img_name]:
            words[img_name][word] += [centroid]
        else:
            words[img_name][word] = [centroid]

        global_y.append(coordenada_y)
        global_x.append(coordenada_x)

    # Interval calculation of the text boxes
    intervalo_y = []
    intervalo_x = []
    for j in range(0, len(global_y)):
        intervalo_y.append([int(max(global_y[j])), int(min(global_y[j]))])
        intervalo_x.append([int(max(global_x[j])), int(min(global_x[j]))])

    # Conversion to grey Scale
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    gauss = cv2.GaussianBlur(gris, (5, 5), 0)

    # Border detection with Canny
    canny = cv2.Canny(gauss, 50, 150)

    # Countour search in the image
    (contornos, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of GUI components detected: ", len(contornos), "\n")

    # We carry out the crops for each detected countour
    shapes = []
    lista_para_no_recortar_dos_veces_mismo_gui = []

    for j in range(0, len(contornos)):
        cont_horizontal = []
        cont_vertical = []
        # Obtain x and y max and min values of the countour
        for i in range(0, len(contornos[j])):
            cont_horizontal.append(contornos[j][i][0][0])
            cont_vertical.append(contornos[j][i][0][1])
        x = min(cont_horizontal)
        x2 = max(cont_horizontal)
        y = min(cont_vertical)
        y2 = max(cont_vertical)

        # Check that the countours are not overlapping with text boxes. If so, cut the text boxes
        condicion_recorte = True

        for k in range(0, len(intervalo_y)):
            solapa_y = 0
            solapa_x = 0
            y_min = min(intervalo_y[k])-cropping_threshold
            y_max = max(intervalo_y[k])+cropping_threshold
            x_min = min(intervalo_x[k])-cropping_threshold
            x_max = max(intervalo_x[k])+cropping_threshold
            solapa_y = (y_min <= y <= y_max) or (y_min <= y2 <= y_max)
            solapa_x = (x_min <= x <= x_max) or (x_min <= x2 <= x_max)
            if (solapa_y and solapa_x):
                if (lista_para_no_recortar_dos_veces_mismo_gui.count(k) == 0):
                    lista_para_no_recortar_dos_veces_mismo_gui.append(k)
                else:
                    # print("Text inside GUI component " + str(k) + " twice")
                    condicion_recorte = False
                x = min(intervalo_x[k])
                x2 = max(intervalo_x[k])
                y = min(intervalo_y[k])
                y2 = max(intervalo_y[k])

        coincidence_with_attention_point = True

        if (condicion_recorte and coincidence_with_attention_point):
            points = [
                [float(x), float(y)],
                [float(x2), float(y)],
                [float(x2), float(y2)],
                [float(x), float(y2)],
            ]

            try:
                if Polygon(points).is_valid == False:
                    continue
            except:
                continue

            shapes.append({
                "label": "Compo",
                "points": points,
            })
        
    for i, shape in enumerate(shapes):
        shape["id"] = i

    return shapes

def get_ocr_image(img_path):
    """
    Applies Keras-OCR over the input image or images to extract plain text and the coordinates corresponding
    to the present words

    :param img_path: img to run ocr over
    :type img_path: str
    :returns: List of lists corresponding the words identified in the input. Example: ('delete', array([[1161.25,  390.  ], [1216.25,  390.  ], [1216.25,  408.75], [1161.25,  408.75]], dtype=float32))
    :rtype: list
    """

    # Get a set of three example images
    image_ocr = keras_ocr.tools.read(img_path)
    
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([image_ocr])

    return prediction_groups