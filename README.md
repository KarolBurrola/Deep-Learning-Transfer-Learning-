# 🐾 Clasificación de 90 especies de animales mediante Transfer Learning (MobileNetV2) y Fine-Tuning 🐾
# Alumna: Karol Burrola Torres

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Cluster](https://img.shields.io/badge/Entrenado%20en-Clúster%20YUCA%20(UNISON)-005A9C)](http://acarus.uson.mx/)
[![Netlify](https://img.shields.io/badge/Despliegue-Netlify-00C7B7?logo=netlify&logoColor=white)](https://superb-marzipan-f3e0b5.netlify.app)

Proyecto de *Redes Neuronales* para clasificar **90 especies diferentes del reino animal** utilizando **Transferencia de Aprendizaje** con la arquitectura **MobileNetV2**. El entrenamiento se llevó a cabo en el **Clúster YUCA** y el modelo resultante fue desplegado en la web mediante **TensorFlow.js** y optimizado para dispositivos móviles con **TensorFlow Lite**.

---

## Enlaces del Proyecto

* 🌐 **Aplicación Web en Vivo (Netlify):** [https://superb-marzipan-f3e0b5.netlify.app](https://superb-marzipan-f3e0b5.netlify.app)
* 📖 **Artículo explicativo en Medium:** [...]

---

## Dataset y Preprocesamiento

* **Fuente:** [Animal Image Dataset (90 Different Animals) en Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).
* **Volumen de datos:** 5,399 imágenes fotográficas reales distribuidas de forma balanceada (~60 fotos por clase).
* **Dataset Estratificado:**
* 🟢 **Entrenamiento (70%):** 3,779 imágenes (Promedio: 42 fotos/clase).
* 🟡 **Validación (15%):** 810 imágenes (Promedio: 9 fotos/clase).
* 🔵 **Prueba / Test (15%):** 810 imágenes (Promedio: 9 fotos/clase).

![Muestra de Clases](outputs/muestra_clases_animales.png)

---

## Arquitectura y Metodología de Entrenamiento

Se utilizó **MobileNetV2** preentrenada en ImageNet, combinada con una cabeza densa personalizada:

1. **Aumento de Datos (*Data Augmentation*):** Capas de `RandomFlip`, `RandomRotation(0.15)`, `RandomZoom(0.1)` y `RandomContrast(0.1)` para mitigar el sobreajuste.
2. **Reescalado de Píxeles:** Normalización matemática al intervalo $[-1, 1]$ mediante `preprocess_input`.
3. **Cabeza de Clasificación:**
   * `GlobalAveragePooling2D` (reducción espacial a 1280 dimensiones).
   * `BatchNormalization` y `Dropout(0.3)`.
   * Capa densa intermedia `Dense(256)` con activación ReLU, regularización L2 (`1e-4`) e inicialización HeNormal.
   * `BatchNormalization` y `Dropout(0.2)`.
   * Capa de salida `Dense(90, activation='softmax')` con inicialización GlorotNormal.

### Fases de Entrenamiento:

* **Fase 1: Extracción de Características (*Feature Extraction*):**  
  Base convolucional completamente congelada. Optimizador Adam con tasa de aprendizaje estándar ($\alpha = 10^{-3}$), función de pérdida `sparse_categorical_crossentropy` y callback `EarlyStopping` con paciencia de 4 épocas.
* **Fase 2: Ajuste Fino (*Fine-Tuning*):**  
  Descongelado parcial de las últimas 54 capas convolucionales (a partir de la capa 100). Optimizador Adam con una tasa de aprendizaje reducida ($\alpha = 10^{-5}$) para no destruir los pesos preentrenados y afinar detalles específicos de las especies.

---

## Resultados
Los scripts generan por ejecución en la carpeta `outputs/`:
* `grafica_acc_animales.png` — Curva de Accuracy con la métrica del inicio de Fine-Tuning 
* `grafica_loss_animales.png` — Curva de Pérdida con la métrica del inicio de Fine-Tuning 
* `matriz_de_confusion_animales.png` — Matriz de confusión de 90x90 clases evaluada sobre las 810 imágenes del conjunto de test.
* `muestra_clases_animales.png` — Muestra de 15 clases de ejemplo del dataset.
* `animals_model.tflite` — Modelo optimizado para celulares.

### Métricas de Desempeño:
| Fase / Conjunto | Época | Accuracy (Entrenamiento) | Pérdida (Loss) | Val Accuracy | Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fase 1: Feature Extraction (Inicio)** | 1 | 40.33 % | 2.9156 | 78.02 % | 1.4302 |
| **Fase 1: Feature Extraction (Final)** | 12 | 90.55 % | 0.4028 | 88.40 % | 0.5033 |
| **Fase 2: Fine-Tuning (Mejor Época)** | 13 | 84.70 % | 0.6145 | **88.52 %** | **0.4974** |
| **Fase 2: Fine-Tuning (Parada Temprana)** | 17 | 89.28 % | 0.4701 | 87.53 % | 0.4983 |
| **Evaluación Final en Test** | — | — | — | **89.75 %** | **0.4811** |
