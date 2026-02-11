# Food Multi-Label Classification Pipeline

<p align="center">
  <img src="https://www.rededucom.org/img/proyectos/14/galeria/big/universidad-salesiana.jpg" width="200">
</p>

<p align="center">
  <strong>Multilabel Dataset Builder + MobileNetV2 Training + MLflow Tracking + Flask Web UI + Incremental Retraining</strong>
</p>

---

## 2. Resumen

Este proyecto implementa un **pipeline completo** para la clasificación multietiqueta (multilabel) de imágenes de alimentos con capacidades de **reentrenamiento incremental**. El sistema genera un dataset sintético realista combinando imágenes de Food-11 y FoodOrNot, entrena un modelo de Deep Learning basado en MobileNetV2, y despliega una interfaz web interactiva que permite a los usuarios corregir predicciones erróneas. Estas correcciones alimentan un bucle de aprendizaje continuo que mejora el modelo automáticamente.


**Objetivos principales:**
1.  Clasificación precisa de múltiples categorías en una sola imagen (Lácteos, Arroz, Frutas/Verduras).
2.  Rechazo de imágenes no relacionadas con alimentos.
3.  Mejora continua del modelo mediante feedback del usuario (Human-in-the-loop).

---

## 3. Stack de Librerías y Tecnologías

El proyecto utiliza un stack moderno de Python para Deep Learning y desarrollo web:

| Categoría | Tecnología/Librería | Propósito |
|-----------|---------------------|-----------|
| **Deep Learning** | `tensorflow`, `keras` | Framework principal (MobileNetV2, tf.data, entrenamiento) |
| **Visión por Computador** | `pillow`, `numpy` | Procesamiento de imágenes y generación de collages |
| **Datos** | `pandas`, `scikit-learn` | Manipulación de datos CSV y división Train/Test |
| **MLOps** | `mlflow` | Rastreo de experimentos, métricas y versionado de modelos |
| **Backend/Web** | `flask` | API REST y servidor web para la interfaz de usuario |
| **Frontend** | HTML5, CSS3, JS | Interfaz de usuario para carga de imágenes y correcciones |

**Instalación de dependencias:**
```bash
pip install tensorflow pandas numpy pillow scikit-learn mlflow flask
```

---

## 4. Método

La metodología sigue un enfoque secuencial de 5 fases, desde la preparación de datos hasta el despliegue y mantenimiento.

### Diagrama del Método

```mermaid
graph TD
    subgraph "Fase 1: Datos"
    A[Datasets Crudos<br>Food11 + FoodOrNot] --> B[Generador de Collages]
    B --> C[Dataset Multilabel<br>(Imágenes + CSV)]
    end

    subgraph "Fase 2: Entrenamiento"
    C --> D[Entrenamiento MobileNetV2<br>(Transfer Learning)]
    D --> E[Evaluación y Tracking<br>(MLflow)]
    end

    subgraph "Fase 3: Despliegue"
    E --> F[Modelo Final (.keras)]
    F --> G[Aplicación Flask]
    end

    subgraph "Fase 4: Feedback Loop"
    G --> H[Predicción Usuario]
    H --> I{¿Correcto?}
    I -- No --> J[Corrección Manual]
    J --> K[Acumulador de Errores]
    K --> L{> 10 Correcciones?}
    L -- Sí --> M[Reentrenamiento Incremental]
    M --> F
    end
```

### Fases Detalladas

1.  **Preparación del Dataset (Data Synthesis)**
    *   Se utilizan imágenes base de *Food-11* (categorías positivas) y *FoodOrNot* (negativos).
    *   Se generan **collages sintéticos** (2x2 grid) para simular platos complejos conformados por múltiples ingredientes.
    *   Etiquetado automático basado en la composición del collage.

2.  **Entrenamiento del Modelo**
    *   Arquitectura: **MobileNetV2** (pre-entrenada en ImageNet) como extractor de características.
    *   Cabezal de clasificación: Capas densas con activación **Sigmoid** para permitir múltiples salidas independientes (multilabel).
    *   Estrategia de entrenamiento en dos etapas: "Head-only" (congelado) seguido de "Fine-tuning" (descongelado parcial).

3.  **Seguimiento de Experimentos**
    *   Uso de **MLflow** para registrar hiperparámetros, métricas (Accuracy, AUC, Loss) y artefactos del modelo.

4.  **Despliegue (Web App)**
    *   Aplicación Flask que sirve el modelo para inferencia en tiempo real.
    *   Interfaz amigable para subir imágenes y visualizar probabilidades por clase.

5.  **Mejora Continua (Incremental Learning)**
    *   Sistema automatizado que recoge correcciones de los usuarios.
    *   Disparador automático de reentrenamiento (fine-tuning conservador) al acumular suficientes nuevos ejemplos.

---

## 5. Diseño de Experimentos

### Configuración del Dataset

Para validar la eficacia del modelo en escenarios complejos, se diseñó un dataset balanceado con las siguientes características:

*   **Fuentes**: Kaggle Food-11, Kaggle FoodOrNot.
*   **Volumen Total**: ~5,800 imágenes generadas (collages 224x224px).
*   **Distribución**:
    *   Entrenamiento: 3,000 imágenes.
    *   Validación: 600 imágenes.
    *   Prueba (Test): 600 imágenes.
*   **Balance de Clases**: Probabilidad de inclusión del 60% por clase para asegurar co-ocurrencias.
*   **Negativos**: 1,200 muestras de "no comida" para robustez.

### Configuración del Modelo (MobileNetV2)

El entrenamiento se realizó bajo las siguientes condiciones experimentales:

*   **Input**: 224x224x3 (RGB).
*   **Optimizador**: Adam.
*   **Loss Function**: Binary Cross-Entropy (estándar para multilabel).
*   **Hiperparámetros**:
    *   *Etapa 1 (Head)*: LR=1e-3, Backbone congelado.
    *   *Etapa 2 (Fine-tune)*: LR=1e-4, Últimas 30 capas descongeladas.
    *   *Dropout*: 0.2 para regularización.

---

## 6. Resultados

El modelo demostró un alto rendimiento en el conjunto de validación, confirmando la viabilidad de MobileNetV2 para tareas multietiqueta en dispositivos con recursos limitados.

### Métricas de Rendimiento

| Etapa | Epoch | Train Accuracy | Val Accuracy | Val AUC | Val Loss |
|-------|-------|----------------|--------------|---------|----------|
| Inicio | 1 | 82.47% | 88.85% | 0.9606 | 0.2702 |
| Final | 10 | **96.19%** | **95.43%** | **0.9915** | **0.1236** |

### Ejemplo de Predicción

El sistema es capaz de desglosar el contenido de una imagen en probabilidades independientes:

> **Imagen de Prueba**: `test_sample.jpg` (Contiene lácteos y frutas)
> *   Lácteos: **92.94%** (Detectado)
> *   Arroz: 2.33% (No detectado)
> *   Frutas/Verduras: **99.95%** (Detectado)
>
> **Resultado Final**: `[Lácteos, Frutas/Verduras]`

---

## 7. Conclusiones

1.  **Eficacia Multietiqueta**: El enfoque multilabel con activación Sigmoid supera a la clasificación tradicional para análisis de alimentos, permitiendo identificar combinaciones de ingredientes realistas (ej. arroz con verduras) con una precisión superior al 95%.
2.  **Aprendizaje Continuo**: La implementación de un bucle de retroalimentación (Feedback Loop) permite al modelo adaptarse a nuevos datos y corregir errores específicos del dominio sin necesidad de reentrenamientos masivos desde cero.
3.  **Eficiencia**: MobileNetV2 ofrece un excelente balance entre precisión y costo computacional, resultando en un modelo ligero (~8.6 MB) apto para despliegue en servidores web modestos o dispositivos Edge.
4.  **Valor Educativo**: Este proyecto integra múltiples disciplinas (Data Engineering, Deep Learning, MLOps, Web Dev), sirviendo como un caso de estudio completo para estudiantes e ingenieros de ML.

---

## 8. Autores

*   **Erika Contreras**
*   **Alexander Chuquipoma**

**Institución:** Universidad Politécnica Salesiana

---

## 9. Información de Contacto

Para consultas sobre el proyecto, colaboración o acceso al código fuente, por favor contactar a:

*   **Email**: [correo_placeholder]@est.ups.edu.ec
*   **Repositorio**: [Enlace al repositorio GitHub]

---
