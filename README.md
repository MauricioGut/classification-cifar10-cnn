# 🧠 Clasificación de Imágenes con CNN y CIFAR-10

Este proyecto implementa una red neuronal convolucional (CNN) en PyTorch para la **clasificación de imágenes del dataset CIFAR-10**, utilizando buenas prácticas en la organización del código y documentación. Forma parte de mi portfolio como estudiante de Ingeniería en Inteligencia Artificial en la UNL.


## 🗂️ Estructura del proyecto

classification-cifar10-cnn/ 
│ 
├── models/     # Modelos entrenados (.pth) 
├── images/     # Gráficos como curva de pérdida 
├── data/       # Dataset CIFAR-10 (descarga automática) 
├── src/ 
│   ├── train.py        # Script de entrenamiento 
│   └── utils.py        # Carga de datos y funciones auxiliares 
├── requirements.txt    # Dependencias 
├── README.md           # Este archivo 
└── cifar10_cnn.ipynb   # (opcional) Notebook alternativo


## ⚙️ Requisitos

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

Instalar con:

pip install -r requirements.txt


## 🚀 Entrenamiento

Ejecutar el script de entrenamiento con:

    python src/train.py

Esto entrena una CNN simple por 10 épocas y guarda:

- El modelo entrenado en `models/cifar10_cnn.pth`
- La curva de pérdida en `images/loss_curve.png`


## 📈 Resultados

Modelo entrenado sobre 10 épocas con validación visual de pérdida.



## 🔮 Posibles mejoras

- Implementar test.py con evaluación cuantitativa

- Agregar visualización de predicciones correctas vs incorrectas

- Comparar con arquitecturas más profundas (ResNet, etc.)

- Añadir notebook interactivo para explicar cada parte paso a paso


## 📚 Créditos

📘 Dataset: CIFAR-10

📘 Base técnica: PyTorch, torchvision

📕 Libros de estudio:
   - Machine Learning for Beginners
   - Linear Algebra and Optimization for Machine Learning – Charu Aggarwal


Este proyecto fue desarrollado como parte de mi formación en inteligencia artificial con enfoque en aprendizaje automático y visión computacional.

---