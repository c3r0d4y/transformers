# Detección de correos de phishing con Transformers

Versión ejecutable del notebook `01_pipeline_completo.ipynb`
(Proyecto Integrador — Módulo Transformers, Diplomado en Inteligencia Artificial,
Centro de Investigación en Computación, IPN).

Compara un baseline clásico contra DistilBERT ajustado por fine-tuning completo y por
LoRA, mide el efecto de la cuantización INT8 y prueba qué tanto generaliza el modelo a
correos en español.

---

## Requisitos

| | |
|---|---|
| **Python** | 3.10 – 3.13 (probado en 3.12) |
| **Sistema** | Linux o macOS. En Windows, usar WSL2 |
| **RAM** | 8 GB mínimo |
| **GPU** | Opcional pero muy recomendable. Con 6 GB de VRAM basta |
| **Disco** | ~4 GB (modelos entrenados, corpus y caché de Hugging Face) |
| **Red** | Necesaria la primera vez: descarga el corpus y los pesos preentrenados |
| **Otros** | `git`, y opcionalmente [`uv`](https://github.com/astral-sh/uv) para instalar más rápido |

Sin GPU el pipeline funciona igual, pero el fine-tuning pasa de ~20 minutos a varias
horas. Para probar que todo corre, usa `--muestra 2000 --epocas 1`.

### Dependencias

Todas están en `requirements.txt` y las instala `setup.sh`:

| Librería | Versión mínima | Para qué |
|---|---|---|
| `torch` | 2.4 | El motor de redes neuronales |
| `transformers` | 4.40 | Modelos preentrenados, tokenizadores y el `Trainer` |
| `datasets` | 2.19 | Descarga y manejo del corpus |
| `peft` | 0.11 | Implementación de LoRA |
| `accelerate` | 0.30 | Requisito del `Trainer` para manejar el dispositivo |
| `scikit-learn` | 1.3 | Baseline TF-IDF y todas las métricas |
| `pandas` | 2.0 | Manipulación de tablas |
| `numpy` | 1.26 | Operaciones numéricas |
| `matplotlib` | 3.7 | Figuras |
| `joblib` | 1.3 | Guardar el modelo del baseline |

Versiones exactas con las que se produjeron los resultados de este repositorio:
Python 3.12.13, torch 2.13.0+cu126, transformers 4.57.6, datasets 4.8.5, peft 0.20.0,
scikit-learn 1.9.0, sobre una NVIDIA Quadro P4000 de 8 GB. Cada corrida deja las suyas
en `results/entorno.json`.

---

## Instalación

```bash
git clone https://github.com/c3r0d4y/transformers.git
cd transformers
./setup.sh                 # crea .venv e instala todo
source .venv/bin/activate
```

`setup.sh` detecta si hay GPU y elige la rueda de PyTorch adecuada. Usa `uv` si está
instalado; si no, el módulo `venv` estándar. Para forzar la versión de solo CPU:

```bash
./setup.sh --cpu
```

### Instalación manual

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **GPUs Pascal (GTX 10xx, Quadro P-series).** Las ruedas de PyTorch publicadas en PyPI
> se compilan contra CUDA reciente y ya no incluyen código para capacidad de cómputo
> 6.x. Si tu tarjeta es de esa generación, instala torch antes que el resto:
> `pip install torch --index-url https://download.pytorch.org/whl/cu126`

---

## Uso

```bash
./pipeline_phishing.py                    # pipeline completo
./pipeline_phishing.py --help             # todas las opciones
```

No hace falta activar el entorno: si faltan las dependencias, el script se reejecuta
solo con el intérprete de `.venv`. Activarlo sigue siendo útil para `generar_es_test.py`
y para trabajar en la consola.

La primera ejecución descarga el corpus (~18,650 correos) y los pesos de
`distilbert-base-uncased` desde Hugging Face, así que necesita conexión a internet.

### Correr por partes

Cada etapa guarda sus resultados en disco, así que el pipeline se puede cortar y retomar:

```bash
./pipeline_phishing.py --etapas datos,baseline
./pipeline_phishing.py --etapas finetune,evaluacion
./pipeline_phishing.py --etapas lora,cuantizacion,reporte
```

| Etapa | Qué hace |
|---|---|
| `datos` | Descarga el corpus, lo limpia, quita duplicados y lo parte en 70/15/15 |
| `baseline` | TF-IDF + Regresión Logística |
| `finetune` | Fine-tuning completo de DistilBERT |
| `evaluacion` | Matrices de confusión, curva precisión-recall y análisis de errores |
| `lora` | Fine-tuning eficiente con LoRA |
| `cuantizacion` | INT8 dinámico en CPU: tamaño, latencia y pérdida de F1 |
| `espanol` | Evalúa el modelo (entrenado en inglés) sobre correos en español |
| `beto` | Entrena BETO con esos correos en español (opcional, hay que pedirla) |
| `reporte` | Tabla comparativa, gráfica final y registro del entorno |
| `demo` | Clasifica textos de ejemplo o los que le pases |

`--etapas todas` (el valor por defecto) corre todo excepto `beto`, porque esa etapa
necesita que el archivo de correos en español ya esté escrito.

### Prueba rápida

Para verificar que todo funciona sin esperar el entrenamiento completo:

```bash
./pipeline_phishing.py --muestra 2000 --epocas 1 --epocas-lora 1
```

### Clasificar texto nuevo

```bash
./pipeline_phishing.py --etapas demo \
  --texto "Your account has been suspended, verify at http://secure-login.example"
./pipeline_phishing.py --etapas demo --archivo-texto correos.txt   # uno por línea
```

### Las gráficas

Todas las figuras se guardan como PNG a 150 dpi en `figures/` (o en
`<--salida>/figures/` si mandas la corrida a otro directorio). El script no abre
ninguna ventana, así que también funciona por SSH o en un servidor sin pantalla.

| Archivo | Etapa que lo genera | Qué muestra |
|---|---|---|
| `01_longitudes.png` | `datos` | Distribución de longitudes y dónde corta `max_length` |
| `02_cm_baseline.png` | `evaluacion` | Matriz de confusión del baseline |
| `03_cm_distilbert.png` | `evaluacion` | Matriz de confusión de DistilBERT |
| `04_curva_pr.png` | `evaluacion` | Curva precisión-recall de ambos modelos |
| `05_cm_espanol.png` | `espanol` | Matriz de confusión sobre los correos en español |
| `06_comparativa.png` | `reporte` | Barras de F1 y recall por modelo |

Dos dependencias entre etapas: `05` solo aparece si corres `espanol`, y `02` necesita
que `baseline` se haya corrido antes que `evaluacion` — si no, la curva PR sale
únicamente con DistilBERT.

Para abrirlas al terminar:

```bash
xdg-open figures/                     # el gestor de archivos
xdg-open figures/06_comparativa.png   # una en particular
```

Si prefieres verlas en ventana conforme se generan, existe `--mostrar`. Ten en cuenta
que cada ventana **detiene el pipeline** hasta que la cierras, así que no la combines
con una corrida larga desatendida:

```bash
./pipeline_phishing.py --etapas evaluacion --mostrar
```

### Opciones útiles

| Opción | Para qué |
|---|---|
| `--salida DIR` | Escribe `data/`, `models/`, `results/` y `figures/` en otro directorio |
| `--forzar` | Rehace etapas aunque ya existan sus resultados |
| `--csv ARCHIVO` | Usa un corpus local en vez de descargarlo de Hugging Face |
| `--muestra N` | Trabaja con solo N correos |
| `--mostrar` | Abre cada figura en una ventana, además de guardarla |
| `--modelo NOMBRE` | Otro modelo base de Hugging Face |
| `--etiqueta NOMBRE` | Nombre corto de la corrida; aísla modelos, figuras y resultados |
| `--max-len`, `--epocas`, `--batch` | Hiperparámetros del entrenamiento |
| `--lora-r`, `--lr-lora`, `--epocas-lora` | Hiperparámetros de LoRA |
| `--seed` | Semilla (por defecto 42) |

---

## Qué produce

```
data/raw/corpus_limpio.csv          corpus tras la limpieza
data/processed/{train,val,test}.csv particiones estratificadas
models/                             modelos entrenados
results/comparativa_modelos.csv     tabla principal del reporte
results/resultados.json             métricas de todos los modelos
results/cuantizacion.csv            tamaño / latencia / F1 de FP32 vs INT8
results/errores_falsos_*.csv        errores concretos para el análisis
results/entorno.json                versiones exactas usadas
results/pipeline.log                registro completo de la ejecución
figures/*.png                       las figuras del reporte, en PNG a 150 dpi
```

De todo eso, **el repositorio solo versiona `figures/`**. Los modelos pesan más de 1 GB,
el corpus procesado 96 MB y los resultados se regeneran corriendo el script, así que
`data/`, `models/` y `results/` quedan fuera. Las cifras de la corrida de referencia
están abajo, en Resultados.

---

## Resultados

Corrida sobre el corpus completo (18,650 correos, 17,500 tras quitar duplicados),
3 épocas de fine-tuning y 5 de LoRA, en una Quadro P4000:

| Modelo | F1 | F1-macro | Parámetros entrenados | Tiempo |
|---|---|---|---|---|
| TF-IDF + Regresión Logística | 0.9797 | 0.9838 | 50,000 | 13 s |
| DistilBERT fine-tuning | **0.9804** | 0.9845 | 66,955,010 | 20 min |
| DistilBERT LoRA | 0.9770 | 0.9817 | 739,586 | 25 min |
| mDistilBERT fine-tuning | 0.9784 | 0.9828 | 135,326,210 | 21 min |

Cuantización INT8 del modelo en inglés: 268 MB → 139 MB y 28.4 → 24.4 ms por correo,
a cambio de bajar el F1 de 0.9737 a 0.9622.

Evaluación en los 100 correos en español, todos los modelos entrenados **solo con
inglés** salvo BETO:

| Modelo | Accuracy | F1-macro | ROC-AUC |
|---|---|---|---|
| DistilBERT (inglés) | 0.5000 | 0.3658 | 0.7928 |
| mDistilBERT (multilingüe) | 0.6800 | 0.6799 | 0.7528 |
| BETO (entrenado con 50 correos en español) | 0.9000 | 0.9000 | 0.9616 |

Tres lecturas que conviene no pasar por alto:

1. **El baseline empata con el Transformer** (0.9797 contra 0.9804). Entre los términos
   más pesados que aprendió aparece `2005`, una fecha: el corpus tiene fuga temporal,
   las dos clases provienen de épocas distintas. El Transformer no tiene margen para
   demostrar nada porque la tarea ya está resuelta por atajos.
2. **LoRA entrena el 1.1% de los parámetros y pierde 0.3% de F1.** Tardó más en total
   solo porque son 5 épocas contra 3; por época fue más rápido, 303 s contra 399 s.
3. **Cambiar a un modelo multilingüe casi duplica el F1-macro en español** sin costar
   nada en inglés, pero 0.68 sigue sin servir para producción. No hay atajo: se
   necesitan datos de entrenamiento en español.

---

## La etapa en español

El repositorio ya incluye `data/es_test.csv` con **100 correos sintéticos** en español,
50 de phishing y 50 legítimos, generados por `generar_es_test.py`. Todos los textos son
inventados y las entidades y dominios son ficticios (`.example` es un TLD reservado y no
resoluble), así que el archivo sirve como banco de prueba sin ser una plantilla usable
contra nadie. No contiene correos reales ni datos personales.

Al construirlo se cuidó que las dos clases no se distingan por artefactos superficiales
en vez de por su contenido:

| | phishing | legítimos |
|---|---|---|
| Palabras por correo | 29.1 | 27.2 |
| Con URL | 76% | 28% |
| Con acentuación | 98% | 98% |

Si todos los phishing llevaran enlace y ninguno de los legítimos, o si una clase se
escribiera sin acentos, el modelo aprendería ese atajo. También hay varios legítimos que
*parecen* sospechosos a propósito —un restablecimiento de contraseña que sí solicitaste,
una petición de firma de contrato, un aviso de verificación en dos pasos— porque son los
que producen los errores más interesantes de analizar.

Si prefieres tu propio conjunto, edita las listas de `generar_es_test.py` y vuelve a
correrlo, o sobrescribe el CSV a mano respetando las columnas `texto` y `etiqueta`:

```bash
./generar_es_test.py data/es_test.csv
```

Una vez que el archivo existe, la etapa `beto` (que no entra en `--etapas todas`) entrena
un BERT en español con la mitad de esos correos y lo evalúa con la otra mitad:

```bash
./pipeline_phishing.py --etapas espanol,beto
```

Son 50 ejemplos de entrenamiento: sirven para mostrar la dirección del efecto, no para
sacar conclusiones fuertes. Dilo así en el reporte.

### Por qué falla en español, y cómo se arregla

La etapa `espanol` no se limita a reportar que el resultado es malo: separa **dos causas
distintas**, porque llevan a conclusiones opuestas en el reporte.

- Si el modelo no entendiera nada, el **ROC-AUC sería 0.5**: no sabría ni ordenar los
  correos de más a menos sospechosos.
- Un ROC-AUC alto con accuracy baja significa otra cosa: el orden es bueno, pero el
  umbral de 0.5 quedó en el lugar equivocado porque las probabilidades se corrieron al
  entrar texto fuera de distribución.

El diagnóstico imprime la probabilidad media por clase, cuántos correos caen del lado
del phishing con umbral 0.5, el ROC-AUC, el umbral que maximizaría el F1-macro y la
**fertilidad del tokenizador** (subtokens por palabra). Guarda además
`07_probs_espanol.png` con las dos distribuciones de probabilidad y los dos umbrales.

Ese umbral calibrado es una **cota optimista**: se elige mirando las etiquetas de esos
mismos 100 correos, así que no es una estimación honesta de despliegue. Sirve para
separar "no entiende" de "no calibra", nada más.

La fertilidad es la que apunta a la causa de fondo. Un tokenizador entrenado solo en
inglés gasta casi el doble de subtokens por palabra en español (`verifique` se parte en
`ve ##ri ##fi ##que`), así que el modelo ve fragmentos sin significado. El arreglo real
es cambiar el modelo base por uno multilingüe y volver a entrenar con el mismo corpus:

```bash
./pipeline_phishing.py --modelo distilbert-base-multilingual-cased \
                       --etiqueta mdistilbert \
                       --etapas finetune,evaluacion,espanol
```

`--etiqueta` le da a esa corrida su propia carpeta en `models/`, su propia subcarpeta en
`figures/`, sus propias claves en `resultados.json` y sus propios archivos en `results/`,
así que **no pisa** los resultados del modelo en inglés y los dos aparecen juntos en la
tabla comparativa final.

---

## Notas de reproducibilidad

- La semilla se fija en `random`, `numpy` y `torch`. Aun así, las operaciones de GPU no
  son bit a bit deterministas: espera variaciones pequeñas entre corridas.
- `fp16` se activa solo en GPUs con Tensor Cores (Volta en adelante). En tarjetas Pascal
  no acelera y puede desestabilizar el entrenamiento, así que el script lo desactiva.
- El script tolera los renombres de la API de `transformers`
  (`evaluation_strategy` → `eval_strategy`, `tokenizer` → `processing_class`), así que
  funciona con versiones viejas y nuevas de la librería.
- `results/entorno.json` guarda las versiones exactas de cada corrida.

## Datos y licencia

Este proyecto se distribuye bajo licencia MIT; el texto completo está en
[`LICENSE`](LICENSE).

Corpus: [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset),
copia en Hugging Face del *Phishing Email Detection* de Kaggle, licencia LGPL-3.0.
Es un corpus en inglés de aproximadamente 18,650 correos etiquetados. El corpus se
descarga al correr el pipeline y no se redistribuye en este repositorio, así que su
licencia no se propaga al código.

Este modelo es un trabajo académico. No está calibrado para desplegarse como filtro real
de correo, y la etapa `espanol` muestra por qué: entrenado solo con inglés, su desempeño
cae de forma notable en otro idioma.
