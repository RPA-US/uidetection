# uidetection
UI Hierarchies Detection from Screenshots

Para empezar a trabajar con el repositorio crearemos un entorno virtual basado en python 3.9:

```
python -m virtualenv python=3.9 env
```

Después ejecutaremos este comando para instalar las dependencias:

```
pip install -r requirements.txt
```

En el caso en el que se quiera utilizar la GPU para estos paquetes:
```
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
```
Instalaremos lo siguiente (modificar la versión de CUDA si es necesario, en este caso se indica la version 11.8):

```
pip3 uninstall torch torchvision torchaudio && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Tenemos que tener en cuenta, que será necesario tener el paquete de datos "YOLO_Datasets/" en la misma ruta, para tener datos sobre los que trabajar.