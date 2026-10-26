import sys
import os
import time
import json
import datetime
import zoneinfo
import random
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import io
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")
tiempo_inicial = time.time()
hora_inicio = datetime.datetime.now(zona_hermosillo)
print(f"Inicio de la ejecucion: {hora_inicio.strftime('%Y-%m-%d %H:%M:%S')}")

import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import kagglehub

print("Descargando dataset de animales desde kaggle...")
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
print("Ruta del dataset descargado:", path)

raw_dataset_dir = Path(path)
posibles_carpetas = [p for p in raw_dataset_dir.rglob("*") if p.is_dir() and len([d for d in p.iterdir() if d.is_dir()]) >= 80]
origen_animales = posibles_carpetas[0] if posibles_carpetas else raw_dataset_dir
print(f"Carpeta base: {origen_animales}")

clases_encontradas = sorted([d.name for d in origen_animales.iterdir() if d.is_dir()])
print(f"Total de clases detectadas: {len(clases_encontradas)}")


def datos_limpios_animales(origen_dir, destino_base, split_train=0.70, split_val=0.15, split_test=0.15):
    destino_base = Path(destino_base)
    if destino_base.exists():
        shutil.rmtree(destino_base)

    imagenes_unicas = defaultdict(dict)
    seen_hashes_global = set()

    clases_dirs = sorted([d for d in Path(origen_dir).iterdir() if d.is_dir()])
    for clase_path in clases_dirs:
        clase = clase_path.name
        archivos = [p for ext in ('*.png', '*.jpg', '*.jpeg') for p in clase_path.glob(ext)]
        for img_path in archivos:
            try:
                h = hashlib.md5(img_path.read_bytes()).hexdigest()
                if h not in seen_hashes_global:
                    seen_hashes_global.add(h)
                    imagenes_unicas[clase][h] = img_path
            except Exception as e:
                print(f"Error leyendo {img_path}: {e}")

    print("\nOrganizando directorios e imagenes...")
    total_copiadas = 0
    for clase, hash_dict in imagenes_unicas.items():
        rutas_unicas = list(hash_dict.values())
        random.seed(42)
        random.shuffle(rutas_unicas)
        total = len(rutas_unicas)
        lim1 = int(total * split_train)
        lim2 = int(total * (split_train + split_val))

        splits = {
            'train': rutas_unicas[:lim1],
            'val': rutas_unicas[lim1:lim2],
            'test': rutas_unicas[lim2:]
        }

        for split_name, archivos in splits.items():
            dest_dir = destino_base / split_name / clase
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in archivos:
                shutil.copy(f, dest_dir / Path(f).name)
                total_copiadas += 1

    print(f"Dataset listo: {total_copiadas} imágenes preparadas.")


nuevo_destino = Path("dataset_limpio")
datos_limpios_animales(origen_animales, nuevo_destino, split_train=0.70, split_val=0.15, split_test=0.15)


def verificar_duplicados(directorios):
    hashes = defaultdict(list)
    print("\nVerificando ausencia de duplicados...")
    for path_dir in map(Path, directorios):
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for file_path in path_dir.rglob(ext):
                h = hashlib.md5(file_path.read_bytes()).hexdigest()
                hashes[h].append(str(file_path))
    dups = {h: r for h, r in hashes.items() if len(r) > 1}
    print(f"Se encontraron {len(dups)} duplicados." if dups else "0 duplicados encontrados.")


verificar_duplicados([nuevo_destino / 'train', nuevo_destino / 'val', nuevo_destino / 'test'])

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    nuevo_destino / "train",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    nuevo_destino / "val",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    nuevo_destino / "test",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
n_classes = len(class_names)
print(f"\nClases de Animales ({n_classes}):", class_names[:10], "... [y 80 más]")

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "labels.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(f"{name}\n")

train_labels = np.concatenate([y for x, y in train_ds], axis=0)
unique_elements, count_elements = np.unique(train_labels, return_counts=True)
print(f"Total de muestras de entrenamiento: {count_elements.sum()}")
print(f"Promedio muestras por clase: {count_elements.mean():.1f} (Min: {count_elements.min()}, Max: {count_elements.max()})")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="aumento_de_datos")

fig = plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(augmented_images[0].numpy().astype("uint8"))
        ax.axis("off")
fig.savefig(output_dir / "aumento_de_datos.png", dpi=300, bbox_inches="tight")
plt.close(fig)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

he_init = initializers.HeNormal(seed=42)
glorot_init = initializers.GlorotNormal(seed=42)
l2_reg = regularizers.l2(1e-4)

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(256, kernel_regularizer=l2_reg, kernel_initializer=he_init)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(n_classes, activation='softmax', kernel_initializer=glorot_init)(x)

modelo_animales = tf.keras.Model(inputs=inputs, outputs=outputs, name="Animals_MobileNetV2")
modelo_animales.summary()

print("\n -- Feature extraction -- ")
initial_learning_rate = 1e-3
modelo_animales.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

stop_early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

initial_epochs = 12
history_fase1 = modelo_animales.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds,
    callbacks=[stop_early]
)

print("\n -- Fine-Tuning -- ")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Capas congeladas: {fine_tune_at} | Capas entrenables: {len(base_model.layers) - fine_tune_at}")

fine_tune_lr = 1e-5
modelo_animales.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = len(history_fase1.epoch) + fine_tune_epochs

history_fase2 = modelo_animales.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=len(history_fase1.epoch),
    validation_data=val_ds,
    callbacks=[stop_early]
)


acc = history_fase1.history['accuracy'] + history_fase2.history['accuracy']
val_acc = history_fase1.history['val_accuracy'] + history_fase2.history['val_accuracy']
loss = history_fase1.history['loss'] + history_fase2.history['loss']
val_loss = history_fase1.history['val_loss'] + history_fase2.history['val_loss']
punto_cambio = len(history_fase1.epoch) - 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(acc, label='Accuracy', color='pink', linewidth=2)
ax.plot(val_acc, label='Val_Accuracy', color='aqua', linewidth=2)
ax.plot([punto_cambio, punto_cambio], ax.get_ylim(), '--', color='orange', label='Inicio fine tuning', linewidth=2)
ax.set_title('Accuracy - Modelo de animales', fontsize=14, fontweight='bold')
ax.set_xlabel('Épocas', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.7)
fig.savefig(output_dir / 'grafica_acc_animales.png', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(loss, label='Loss', color='pink', linewidth=2)
ax.plot(val_loss, label='Val_loss', color='aqua', linewidth=2)
ax.plot([punto_cambio, punto_cambio], ax.get_ylim(), '--', color='orange', label='Inicio fine tuning', linewidth=2)
ax.set_title('Pérdida - Modelo de animales', fontsize=14, fontweight='bold')
ax.set_xlabel('Épocas', fontsize=12)
ax.set_ylabel('Pérdida (Cross-Entropy)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.7)
fig.savefig(output_dir / 'grafica_loss_animales.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("\nEvaluando en el conjunto test...")
test_loss, test_acc = modelo_animales.evaluate(test_ds)
print(f"Loss en Test: {test_loss:.4f}")
print(f"Accuracy en Test: {test_acc*100:.2f}%")

y_test = []
y_pred_probs = []
for images, labels in test_ds:
    y_test.extend(labels.numpy())
    predicciones_batch = modelo_animales.predict(images, verbose=0)
    y_pred_probs.extend(predicciones_batch)

pred_test = np.argmax(y_pred_probs, axis=1)

con = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(24, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=con, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=90, colorbar=True)
ax.set_title('Matriz de confusión - Modelo de animales', fontsize=16, fontweight='bold', pad=15)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
fig.savefig(output_dir / 'matriz_de_confusion_animales.png', dpi=300, bbox_inches='tight')
plt.close(fig)

web_dir = output_dir / "web_model"
web_dir.mkdir(parents=True, exist_ok=True)

(web_dir / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
modelo_animales.save(output_dir / "animals_model.h5")

try:
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(modelo_animales, str(web_dir))

    model_json_path = web_dir / "model.json"
    if model_json_path.exists():
        with open(model_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        def clean_layer(l):
            cfg = l.get('config', {})
            for reg_key in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'depthwise_regularizer']:
                if reg_key in cfg:
                    cfg[reg_key] = None
            for init_key in ['kernel_initializer', 'bias_initializer', 'depthwise_initializer', 'gamma_initializer', 'beta_initializer', 'moving_mean_initializer', 'moving_variance_initializer']:
                if init_key in cfg and isinstance(cfg[init_key], dict):
                    cname = cfg[init_key].get('class_name', 'Zeros')
                    cfg[init_key] = {'class_name': cname, 'config': {}}
            for con_key in ['kernel_constraint', 'bias_constraint', 'depthwise_constraint', 'gamma_constraint', 'beta_constraint']:
                if con_key in cfg:
                    cfg[con_key] = None

        def clean_dtype(obj):
            if isinstance(obj, dict):
                if 'dtype' in obj and isinstance(obj['dtype'], dict):
                    obj['dtype'] = 'float32'
                for v in obj.values():
                    clean_dtype(v)
            elif isinstance(obj, list):
                for v in obj:
                    clean_dtype(v)

        def extract_keras_history(arg):
            if isinstance(arg, dict) and arg.get('class_name') == '__keras_tensor__':
                hist = arg.get('config', {}).get('keras_history', [])
                if hist:
                    layer_name = hist[0]
                    if layer_name in ('input_layer_1', 'input_layer_2', 'subtract', 'true_divide', 'aumento_de_datos'):
                        layer_name = 'input_1'
                    return [layer_name, hist[1], hist[2], {}]
            return None

        def convert_inbound(inbound_nodes):
            if not inbound_nodes:
                return []
            res_nodes = []
            for node in inbound_nodes:
                if isinstance(node, dict) and 'args' in node:
                    args = node['args']
                    node_entries = []
                    for a in args:
                        if isinstance(a, list):
                            for sub_a in a:
                                entry = extract_keras_history(sub_a)
                                if entry:
                                    node_entries.append(entry)
                        else:
                            entry = extract_keras_history(a)
                            if entry:
                                node_entries.append(entry)
                    if node_entries:
                        res_nodes.append(node_entries)
                elif isinstance(node, list):
                    res_nodes.append(node)
            return res_nodes

        if 'model_config' in raw_data.get('modelTopology', {}):
            mc = raw_data['modelTopology']['model_config']['config']
            top_layers = mc['layers']
            mobilenet = next(l for l in top_layers if l.get('name') == 'mobilenetv2_1.00_224')
            mb_layers = mobilenet['config']['layers']

            final_layers = [{
                'class_name': 'InputLayer',
                'config': {
                    'batch_input_shape': [None, 224, 224, 3],
                    'dtype': 'float32',
                    'sparse': False,
                    'name': 'input_1'
                },
                'name': 'input_1',
                'inbound_nodes': []
            }]

            dw_layer_names = set()
            for l in mb_layers:
                if l.get('class_name') == 'InputLayer':
                    continue
                if l.get('class_name') == 'DepthwiseConv2D':
                    dw_layer_names.add(l.get('name'))
                c = l.copy()
                c['inbound_nodes'] = convert_inbound(c.get('inbound_nodes', []))
                clean_dtype(c)
                clean_layer(c)
                final_layers.append(c)

            found_gap = False
            for l in top_layers:
                if l.get('class_name') == 'GlobalAveragePooling2D':
                    found_gap = True
                if found_gap:
                    c = l.copy()
                    c['inbound_nodes'] = convert_inbound(c.get('inbound_nodes', []))
                    if l.get('class_name') == 'GlobalAveragePooling2D':
                        c['inbound_nodes'] = [[['out_relu', 0, 0, {}]]]
                    clean_dtype(c)
                    clean_layer(c)
                    final_layers.append(c)

            for m in raw_data['weightsManifest']:
                for w in m['weights']:
                    p = w['name'].split('/')[0]
                    if p in dw_layer_names and w['name'].endswith('/kernel'):
                        w['name'] = p + '/depthwise_kernel'

            fixed_json = {
                'format': 'layers-model',
                'generatedBy': 'keras v2.15.0',
                'convertedBy': 'TensorFlow.js Converter v4.22.0',
                'modelTopology': {
                    'keras_version': '2.15.0',
                    'backend': 'tensorflow',
                    'class_name': 'Functional',
                    'config': {
                        'name': 'Animals_MobileNetV2',
                        'layers': final_layers,
                        'input_layers': [['input_1', 0, 0]],
                        'output_layers': [['dense_1', 0, 0]]
                    }
                },
                'weightsManifest': raw_data['weightsManifest']
            }
            with open(model_json_path, "w", encoding="utf-8") as f:
                json.dump(fixed_json, f, indent=2)

        shutil.make_archive(str(output_dir / "web_model"), "zip", str(web_dir))
except Exception:
    pass

try:
    with contextlib.redirect_stdout(io.StringIO()):
        converter = tf.lite.TFLiteConverter.from_keras_model(modelo_animales)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

    with open(output_dir / "animals_model.tflite", "wb") as f:
        f.write(tflite_model)
except Exception:
    pass

fig = plt.figure(figsize=(15, 9))
muestras_clases = random.sample(class_names, min(15, len(class_names)))
for idx, clase in enumerate(muestras_clases):
    clase_dir = nuevo_destino / "train" / clase
    img_path = next(p for p in clase_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    ax = fig.add_subplot(3, 5, idx + 1)
    ax.imshow(img)
    ax.set_title(clase, fontsize=11, fontweight="bold")
    ax.axis("off")

fig.suptitle("Muestra de 15 especies de animales", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "muestra_clases_animales.png", dpi=300, bbox_inches="tight")
plt.close(fig)

hora_final = datetime.datetime.now(zona_hermosillo)
print(f"\nHermosillo, {hora_final.hour}:{hora_final.minute}:{hora_final.second}")
print(f"Tiempo total: {(time.time() - tiempo_inicial)/60:.5f} mins")