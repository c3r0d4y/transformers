#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deteccion de correos de phishing con Transformers.

Version ejecutable del notebook 01_pipeline_completo.ipynb
(Proyecto Integrador - Modulo Transformers, Diplomado en IA, CIC-IPN).

El pipeline se divide en etapas independientes que guardan sus resultados en
disco, asi que puede correrse completo o por partes:

    datos         descarga, limpieza y particion del corpus
    baseline      TF-IDF + Regresion Logistica
    finetune      fine-tuning completo de DistilBERT
    evaluacion    matrices de confusion, curva PR y analisis de errores
    lora          fine-tuning eficiente con LoRA
    cuantizacion  INT8 dinamico en CPU (tamano / latencia / F1)
    espanol       generalizacion a correos en espanol
    beto          entrenamiento de BETO con los correos en espanol (opcional)
    reporte       tabla comparativa, grafica final y registro del entorno
    demo          clasificacion de textos de ejemplo

Ejemplos:
    ./pipeline_phishing.py                              # todo el pipeline
    ./pipeline_phishing.py --etapas datos,baseline      # solo hasta el baseline
    ./pipeline_phishing.py --muestra 2000 --epocas 1    # prueba rapida
    ./pipeline_phishing.py --etapas demo --texto "Verify your account now"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path


def _saltar_al_venv() -> None:
    """Reejecuta el script con el .venv del proyecto si falta alguna dependencia.

    Asi './pipeline_phishing.py' funciona aunque el usuario no haya corrido
    'source .venv/bin/activate' antes. Si no hay .venv, no hace nada y el
    error de importacion normal se encarga de avisar.
    """
    if os.environ.get("PIPELINE_SIN_REEXEC"):
        return

    proyecto = Path(__file__).resolve().parent
    interprete = proyecto / ".venv" / "bin" / "python"
    if not interprete.exists() or Path(sys.executable).resolve() == interprete:
        return

    from importlib.util import find_spec
    if all(find_spec(m) is not None
           for m in ("matplotlib", "numpy", "pandas", "torch", "transformers")):
        return

    print(f"Usando el entorno del proyecto: {interprete}", file=sys.stderr)
    os.environ["PIPELINE_SIN_REEXEC"] = "1"
    os.execv(str(interprete), [str(interprete), str(Path(__file__).resolve()), *sys.argv[1:]])


_saltar_al_venv()

# Backend sin ventana grafica: el script debe correr en servidores sin pantalla.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------

CORPUS_HF = "zefang-liu/phishing-email-dataset"
MODELO_EN = "distilbert-base-uncased"
MODELO_ES = "dccuchile/bert-base-spanish-wwm-cased"

ETAPAS = ["datos", "baseline", "finetune", "evaluacion", "lora",
          "cuantizacion", "espanol", "beto", "reporte", "demo"]

# Etapas que se ejecutan cuando no se pide nada en particular.
# BETO queda fuera porque necesita los 100 correos en espanol ya escritos.
ETAPAS_POR_DEFECTO = [e for e in ETAPAS if e != "beto"]

ETIQUETAS = ["legitimo", "phishing"]

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

class Rutas:
    """Todas las rutas del proyecto, colgando de un directorio de salida."""

    def __init__(self, base: Path):
        self.base = base
        self.raw = base / "data" / "raw"
        self.processed = base / "data" / "processed"
        self.datos = base / "data"
        self.results = base / "results"
        self.figures = base / "figures"
        self.models = base / "models"

    def crear(self) -> None:
        for d in (self.raw, self.processed, self.results, self.figures, self.models):
            d.mkdir(parents=True, exist_ok=True)


def configurar_log(archivo: Path) -> None:
    archivo.parent.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    consola = logging.StreamHandler(sys.stdout)
    consola.setFormatter(formato)

    fichero = logging.FileHandler(archivo, encoding="utf-8")
    fichero.setFormatter(formato)

    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(consola)
    log.addHandler(fichero)


def titulo(texto: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info(texto)
    log.info("=" * 72)


def fijar_semillas(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def usar_fp16() -> bool:
    """fp16 solo en GPUs con Tensor Cores (Volta en adelante).

    En tarjetas Pascal (GTX 10xx, Quadro P-series) la mitad de precision no
    acelera nada y puede provocar overflow en el entrenamiento.
    """
    import torch
    if not torch.cuda.is_available():
        return False
    mayor, _ = torch.cuda.get_device_capability(0)
    return mayor >= 7


def nombres_corrida(args) -> dict:
    """Nombres de carpetas, claves de resultados y figuras para el modelo elegido.

    Permite entrenar varios modelos base en el mismo directorio de salida sin
    que uno pise los resultados del otro. El modelo por defecto conserva los
    nombres originales para no romper las corridas ya hechas.
    """
    etiqueta = args.etiqueta or re.sub(r"[^a-z0-9]+", "_", args.modelo.lower()).strip("_")
    if args.modelo == MODELO_EN and not args.etiqueta:
        return {"etiqueta": "distilbert_ft", "por_defecto": True,
                "dir_ft": "distilbert_ft", "dir_lora": "distilbert_lora",
                "res_ft": "distilbert_finetune", "res_lora": "distilbert_lora",
                "res_es": "distilbert_en_espanol"}
    return {"etiqueta": etiqueta, "por_defecto": False,
            "dir_ft": etiqueta, "dir_lora": f"{etiqueta}_lora",
            "res_ft": f"{etiqueta}_finetune", "res_lora": f"{etiqueta}_lora",
            "res_es": f"{etiqueta}_en_espanol"}


def archivo_corrida(args, rutas: Rutas, nombre: str) -> Path:
    """Ruta en results/ propia de esta corrida, para no pisar la de otro modelo."""
    if args.nombres["por_defecto"]:
        return rutas.results / nombre
    return rutas.results / f"{args.nombres['etiqueta']}_{nombre}"


def cerrar_figura(fig, archivo: Path, mostrar: bool = False) -> None:
    """Guarda la figura en disco y, si se pidio --mostrar, la abre en una ventana.

    plt.show() bloquea hasta que se cierra la ventana: es lo que se quiere al
    revisar los resultados a mano, pero por eso no es el comportamiento normal.
    """
    fig.savefig(archivo, dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)


def guardar_json(obj, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cargar_json(ruta: Path):
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {}


def acumular_resultado(rutas: Rutas, nombre: str, metricas: dict) -> None:
    """Guarda el resultado de un modelo en results/resultados.json.

    Se escribe despues de cada etapa para que el pipeline se pueda cortar y
    retomar sin perder lo ya calculado.
    """
    archivo = rutas.results / "resultados.json"
    acumulado = cargar_json(archivo)
    acumulado[nombre] = metricas
    guardar_json(acumulado, archivo)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def metricas(y_verdadero, y_predicho, y_prob=None) -> dict:
    """Conjunto de metricas que se reporta para todos los modelos."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score)
    r = {
        "accuracy":  accuracy_score(y_verdadero, y_predicho),
        "precision": precision_score(y_verdadero, y_predicho, zero_division=0),
        "recall":    recall_score(y_verdadero, y_predicho, zero_division=0),
        "f1":        f1_score(y_verdadero, y_predicho, zero_division=0),
        "f1_macro":  f1_score(y_verdadero, y_predicho, average="macro", zero_division=0),
    }
    if y_prob is not None and len(set(np.asarray(y_verdadero).tolist())) > 1:
        r["roc_auc"] = roc_auc_score(y_verdadero, y_prob)
        r["pr_auc"] = average_precision_score(y_verdadero, y_prob)
    return {k: float(v) for k, v in r.items()}


def reporte_clasificacion(y_verdadero, y_predicho) -> None:
    from sklearn.metrics import classification_report
    log.info("\n%s", classification_report(y_verdadero, y_predicho,
                                           target_names=ETIQUETAS, digits=4,
                                           zero_division=0))


# ---------------------------------------------------------------------------
# Etapa 1 - Datos
# ---------------------------------------------------------------------------

def detectar_columnas(df_raw: pd.DataFrame) -> tuple[str, str]:
    """Detecta cual columna trae el texto y cual la etiqueta.

    La de texto es la de strings mas largos en promedio; la de etiqueta es la
    que tiene entre 2 y 5 valores distintos. Asi el script no depende de que
    los nombres exactos del corpus no cambien.
    """
    col_texto, col_etiqueta, mayor_largo = None, None, 0.0

    for c in df_raw.columns:
        serie = df_raw[c].dropna()
        if len(serie) == 0 or not pd.api.types.is_string_dtype(serie):
            continue
        n_unicos = serie.nunique()
        largo_medio = float(serie.astype(str).str.len().mean())
        if 2 <= n_unicos <= 5 and col_etiqueta is None:
            col_etiqueta = c
        elif largo_medio > mayor_largo:
            mayor_largo = largo_medio
            col_texto = c

    # Respaldo con los nombres esperados del corpus original
    return col_texto or "Email Text", col_etiqueta or "Email Type"


def a_binario(valor) -> int:
    """1 = phishing (la clase que interesa detectar), 0 = legitimo."""
    v = str(valor).strip().lower()
    return int(any(p in v for p in ("phish", "spam", "fraud", "malicious")))


def etapa_datos(args, rutas: Rutas) -> None:
    titulo("Etapa 1/9 - Descarga, limpieza y particion del corpus")

    destino = rutas.raw / "corpus_limpio.csv"
    particiones = [rutas.processed / f"{n}.csv" for n in ("train", "val", "test")]
    if all(p.exists() for p in particiones) and not args.forzar:
        log.info("Particiones ya existentes en %s (usa --forzar para rehacerlas)",
                 rutas.processed)
        return

    if args.csv:
        log.info("Leyendo corpus local: %s", args.csv)
        df_raw = pd.read_csv(args.csv)
    else:
        from datasets import load_dataset
        log.info("Descargando corpus '%s' de Hugging Face...", CORPUS_HF)
        df_raw = load_dataset(CORPUS_HF)["train"].to_pandas()

    log.info("Filas: %d | Columnas: %s", len(df_raw), list(df_raw.columns))

    col_texto, col_etiqueta = detectar_columnas(df_raw)
    log.info("Columna de texto detectada   : %s", col_texto)
    log.info("Columna de etiqueta detectada: %s", col_etiqueta)
    log.info("Valores de la etiqueta:\n%s", df_raw[col_etiqueta].value_counts())

    df = df_raw[[col_texto, col_etiqueta]].copy()
    df.columns = ["texto", "etiqueta_original"]
    df["etiqueta"] = df["etiqueta_original"].apply(a_binario)

    # --- Limpieza ---------------------------------------------------------
    # El paso critico es quitar duplicados: si un mismo correo cae en train y
    # en test, el modelo lo memoriza y las metricas salen infladas (fuga de
    # informacion). No se pasa a minusculas: 'URGENTE!!!' es senal util.
    antes = len(df)
    df = df.dropna(subset=["texto"])
    df["texto"] = (df["texto"].astype(str)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip())
    df = df[df["texto"].str.len() >= 20]
    df = df[~df["texto"].str.lower().isin(["empty", "none", "nan", "null"])]
    antes_dup = len(df)
    df = df.drop_duplicates(subset=["texto"], keep="first").reset_index(drop=True)

    log.info("Registros iniciales    : %d", antes)
    log.info("Tras limpieza basica   : %d  (eliminados %d)", antes_dup, antes - antes_dup)
    log.info("Tras quitar duplicados : %d  (eliminados %d)", len(df), antes_dup - len(df))
    log.info("Proporcion de phishing : %.1f%%", 100 * df["etiqueta"].mean())

    from sklearn.model_selection import train_test_split

    if args.muestra and args.muestra < len(df):
        # Submuestra estratificada: conserva la proporcion de phishing original
        df, _ = train_test_split(df, train_size=args.muestra,
                                 stratify=df["etiqueta"], random_state=args.seed)
        df = df.reset_index(drop=True)
        log.info("Submuestra de trabajo  : %d correos (--muestra)", len(df))

    # Distribucion de longitudes: justifica el valor de max_length
    n_palabras = df["texto"].str.split().str.len()
    log.info("Palabras por correo:\n%s",
             n_palabras.describe(percentiles=[.5, .75, .9, .95]).round(1))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(n_palabras.clip(upper=1000), bins=60, color="#4C72B0")
    ax.axvline(args.max_len, color="crimson", linestyle="--",
               label=f"max_length = {args.max_len}")
    ax.set_xlabel("Palabras por correo (recortado a 1000)")
    ax.set_ylabel("Cantidad de correos")
    ax.set_title("Distribucion de longitudes")
    ax.legend()
    fig.tight_layout()
    cerrar_figura(fig, rutas.figs / "01_longitudes.png", args.mostrar)

    df[["texto", "etiqueta"]].to_csv(destino, index=False)

    # --- Particion estratificada 70 / 15 / 15 -----------------------------
    train_df, temp_df = train_test_split(
        df[["texto", "etiqueta"]], test_size=0.30,
        stratify=df["etiqueta"], random_state=args.seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df["etiqueta"], random_state=args.seed)

    for nombre, parte in (("train", train_df), ("val", val_df), ("test", test_df)):
        parte.reset_index(drop=True).to_csv(rutas.processed / f"{nombre}.csv", index=False)
        log.info("%-6s: %6d correos | phishing: %.1f%%",
                 nombre, len(parte), 100 * parte["etiqueta"].mean())


def cargar_particiones(rutas: Rutas) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    faltan = [n for n in ("train", "val", "test")
              if not (rutas.processed / f"{n}.csv").exists()]
    if faltan:
        raise SystemExit(
            f"Faltan las particiones {faltan} en {rutas.processed}. "
            "Corre primero la etapa 'datos'.")
    partes = []
    for n in ("train", "val", "test"):
        d = pd.read_csv(rutas.processed / f"{n}.csv")
        d["texto"] = d["texto"].astype(str)
        partes.append(d)
    return tuple(partes)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Etapa 2 - Baseline TF-IDF + Regresion Logistica
# ---------------------------------------------------------------------------

def etapa_baseline(args, rutas: Rutas) -> None:
    titulo("Etapa 2/9 - Baseline: TF-IDF + Regresion Logistica")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import joblib

    train_df, _, test_df = cargar_particiones(rutas)

    t0 = time.time()
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=args.seed)),
    ])
    baseline.fit(train_df["texto"], train_df["etiqueta"])
    tiempo = time.time() - t0

    pred = baseline.predict(test_df["texto"])
    prob = baseline.predict_proba(test_df["texto"])[:, 1]

    m = metricas(test_df["etiqueta"], pred, prob)
    m["tiempo_entrenamiento_s"] = round(tiempo, 1)
    m["parametros_entrenados"] = int(baseline.named_steps["clf"].coef_.size)
    acumular_resultado(rutas, "baseline_tfidf", m)

    log.info("Entrenado en %.1f segundos", tiempo)
    reporte_clasificacion(test_df["etiqueta"], pred)

    # Control de sanidad: si las palabras top no tienen sentido, algo esta mal
    vocab = np.array(baseline.named_steps["tfidf"].get_feature_names_out())
    pesos = baseline.named_steps["clf"].coef_[0]
    log.info("Indicadores mas fuertes de PHISHING:")
    for i in np.argsort(pesos)[-15:][::-1]:
        log.info("   %-28s %+.3f", vocab[i], pesos[i])
    log.info("Indicadores mas fuertes de LEGITIMO:")
    for i in np.argsort(pesos)[:15]:
        log.info("   %-28s %+.3f", vocab[i], pesos[i])

    if m["f1"] > 0.98:
        log.warning("AVISO: el baseline ya supera F1 = 0.98. Revisa las palabras de "
                    "arriba: si son nombres o fragmentos de encabezado, el corpus "
                    "tiene pistas triviales. Considera --muestra 2000.")

    joblib.dump(baseline, rutas.models / "baseline_tfidf.joblib")
    np.save(rutas.results / "pred_baseline.npy", np.c_[pred, prob])


# ---------------------------------------------------------------------------
# Compatibilidad con distintas versiones de transformers
# ---------------------------------------------------------------------------

def construir_args(carpeta: Path, args, epocas=3, lr=2e-5, batch=None):
    """TrainingArguments valido en versiones nuevas y viejas de transformers.

    'evaluation_strategy' se renombro a 'eval_strategy', asi que se prueban
    ambos nombres.
    """
    from transformers import TrainingArguments
    batch = batch or args.batch
    comunes = dict(
        output_dir=str(carpeta),
        num_train_epochs=epocas,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        fp16=usar_fp16(),
        dataloader_num_workers=2,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", save_strategy="epoch", **comunes)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", **comunes)


def metricas_trainer(pred):
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    logits, y = pred
    if isinstance(logits, tuple):
        logits = logits[0]
    y_pred = np.argmax(logits, axis=-1)
    return {"f1": f1_score(y, y_pred, zero_division=0),
            "accuracy": accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred, zero_division=0)}


def crear_trainer(modelo, entrenamiento_args, tokenizer, ds_train, ds_val, collator):
    """El argumento 'tokenizer' del Trainer se renombro a 'processing_class'."""
    from transformers import Trainer
    base = dict(model=modelo, args=entrenamiento_args, train_dataset=ds_train,
                eval_dataset=ds_val, data_collator=collator,
                compute_metrics=metricas_trainer)
    try:
        return Trainer(processing_class=tokenizer, **base)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **base)


def tokenizar(dframe: pd.DataFrame, tokenizer, max_len: int):
    from datasets import Dataset
    d = Dataset.from_pandas(dframe[["texto", "etiqueta"]].reset_index(drop=True))
    d = d.map(lambda lote: tokenizer(lote["texto"], truncation=True, max_length=max_len),
              batched=True, remove_columns=["texto"])
    return d.rename_column("etiqueta", "labels")


def preparar_datos_hf(args, rutas: Rutas):
    """Devuelve (tokenizer, collator, ds_train, ds_val, ds_test, dataframes)."""
    from transformers import AutoTokenizer, DataCollatorWithPadding
    train_df, val_df, test_df = cargar_particiones(rutas)
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = [tokenizar(d, tokenizer, args.max_len) for d in (train_df, val_df, test_df)]
    return tokenizer, collator, ds[0], ds[1], ds[2], (train_df, val_df, test_df)


# ---------------------------------------------------------------------------
# Etapa 3 - Fine-tuning completo
# ---------------------------------------------------------------------------

def etapa_finetune(args, rutas: Rutas) -> None:
    titulo("Etapa 3/9 - Fine-tuning completo de " + args.modelo)

    import torch
    from transformers import AutoModelForSequenceClassification

    destino = rutas.models / args.nombres["dir_ft"] / "final"
    tokenizer, collator, ds_train, ds_val, ds_test, (train_df, _, test_df) = \
        preparar_datos_hf(args, rutas)

    # Cuantos correos se truncan con este max_length (dato para el reporte)
    muestra = train_df["texto"].head(2000)
    # verbose=False evita el aviso de "sequence longer than 512": aqui se mide
    # a proposito la longitud real, sin truncar.
    largos = np.array([len(tokenizer(t, truncation=False, verbose=False)["input_ids"])
                       for t in muestra])
    pct_truncado = float((largos > args.max_len).mean())
    log.info("Correos que superan %d tokens: %.1f%% (anotalo en la seccion de Datos)",
             args.max_len, 100 * pct_truncado)

    entrenamiento_args = construir_args(rutas.models / args.nombres["dir_ft"], args,
                                        epocas=args.epocas)

    if destino.exists() and not args.forzar:
        log.info("Modelo ya entrenado en %s (usa --forzar para reentrenar)", destino)
        modelo = AutoModelForSequenceClassification.from_pretrained(destino)
        tiempo = float(cargar_json(rutas.results / "resultados.json")
                       .get(args.nombres["res_ft"], {}).get("tiempo_entrenamiento_s", 0))
        trainer = crear_trainer(modelo, entrenamiento_args, tokenizer,
                                ds_train, ds_val, collator)
    else:
        modelo = AutoModelForSequenceClassification.from_pretrained(
            args.modelo, num_labels=2,
            id2label={0: "legitimo", 1: "phishing"},
            label2id={"legitimo": 0, "phishing": 1})
        trainer = crear_trainer(modelo, entrenamiento_args, tokenizer,
                                ds_train, ds_val, collator)
        t0 = time.time()
        trainer.train()
        tiempo = time.time() - t0
        log.info("Entrenamiento completo en %.1f minutos", tiempo / 60)
        trainer.save_model(str(destino))
        tokenizer.save_pretrained(str(destino))
        modelo = trainer.model

    # Prediccion sobre el conjunto de prueba (se toca una sola vez, al final)
    salida = trainer.predict(ds_test)
    logits = salida.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, axis=-1)
    prob = softmax_np(logits, axis=-1)[:, 1]

    m = metricas(test_df["etiqueta"], pred, prob)
    m["tiempo_entrenamiento_s"] = round(float(tiempo), 1)
    m["parametros_entrenados"] = int(sum(p.numel() for p in modelo.parameters()
                                         if p.requires_grad))
    m["pct_truncado"] = round(pct_truncado, 4)
    acumular_resultado(rutas, args.nombres["res_ft"], m)

    reporte_clasificacion(test_df["etiqueta"], pred)
    np.save(archivo_corrida(args, rutas, "pred_finetune.npy"), np.c_[pred, prob])


# ---------------------------------------------------------------------------
# Etapa 4 - Evaluacion y analisis de errores
# ---------------------------------------------------------------------------

def graficar_confusion(y_verdadero, y_predicho, titulo_fig: str, archivo: Path,
                       mostrar: bool = False):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_verdadero, y_predicho, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ETIQUETAS)
    ax.set_yticks([0, 1], ETIQUETAS)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    ax.set_title(titulo_fig)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cerrar_figura(fig, archivo, mostrar)
    return cm


def etapa_evaluacion(args, rutas: Rutas) -> None:
    titulo("Etapa 4/9 - Evaluacion y analisis de errores")

    from sklearn.metrics import precision_recall_curve, average_precision_score

    _, _, test_df = cargar_particiones(rutas)
    archivo_base = rutas.results / "pred_baseline.npy"
    archivo_ft = archivo_corrida(args, rutas, "pred_finetune.npy")
    if not archivo_ft.exists():
        log.warning("Falta pred_finetune.npy. Corre la etapa 'finetune' primero.")
        return

    pred_ft, prob_ft = np.load(archivo_ft).T
    pred_ft = pred_ft.astype(int)

    cm_ft = graficar_confusion(test_df["etiqueta"], pred_ft,
                               "DistilBERT fine-tuning",
                               rutas.figs / "03_cm_distilbert.png", args.mostrar)

    curvas = [("DistilBERT", prob_ft)]
    if archivo_base.exists():
        pred_b, prob_b = np.load(archivo_base).T
        cm_b = graficar_confusion(test_df["etiqueta"], pred_b.astype(int),
                                  "Baseline TF-IDF",
                                  rutas.figs / "02_cm_baseline.png", args.mostrar)
        curvas.insert(0, ("Baseline TF-IDF", prob_b))
        log.info("Phishing no detectado (falsos negativos) -> baseline: %d | DistilBERT: %d",
                 cm_b[1, 0], cm_ft[1, 0])
        log.info("Falsas alarmas       (falsos positivos)  -> baseline: %d | DistilBERT: %d",
                 cm_b[0, 1], cm_ft[0, 1])
    else:
        log.info("Falsos negativos DistilBERT: %d | falsos positivos: %d",
                 cm_ft[1, 0], cm_ft[0, 1])

    # Curva precision-recall: el intercambio entre atrapar todo y no molestar
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nombre, probs in curvas:
        p, r, _ = precision_recall_curve(test_df["etiqueta"], probs)
        ap = average_precision_score(test_df["etiqueta"], probs)
        ax.plot(r, p, label=f"{nombre} (PR-AUC = {ap:.4f})")
    ax.set_xlabel("Recall (que porcentaje del phishing atrapo)")
    ax.set_ylabel("Precision (de lo que marque, cuanto era phishing)")
    ax.set_title("Curva precision-recall")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    cerrar_figura(fig, rutas.figs / "04_curva_pr.png", args.mostrar)

    # Los errores concretos: materia prima de la seccion de Analisis del reporte
    ev = test_df.reset_index(drop=True).copy()
    ev["pred"] = pred_ft
    ev["prob_phishing"] = prob_ft

    fn = ev[(ev.etiqueta == 1) & (ev.pred == 0)].sort_values("prob_phishing")
    fp = ev[(ev.etiqueta == 0) & (ev.pred == 1)].sort_values("prob_phishing",
                                                             ascending=False)
    log.info("Falsos negativos: %d | Falsos positivos: %d", len(fn), len(fp))
    log.info("--- Phishing que se colo (los 5 errores mas confiados) ---")
    for _, fila in fn.head(5).iterrows():
        log.info("[prob. phishing = %.3f] %s", fila.prob_phishing, fila.texto[:300])

    fn.head(30).to_csv(archivo_corrida(args, rutas, "errores_falsos_negativos.csv"), index=False)
    fp.head(30).to_csv(archivo_corrida(args, rutas, "errores_falsos_positivos.csv"), index=False)
    log.info("Errores guardados en %s", rutas.results)


# ---------------------------------------------------------------------------
# Etapa 5 - LoRA
# ---------------------------------------------------------------------------

def etapa_lora(args, rutas: Rutas) -> None:
    titulo("Etapa 5/9 - LoRA (fine-tuning eficiente en parametros)")

    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer, collator, ds_train, ds_val, ds_test, (_, _, test_df) = \
        preparar_datos_hf(args, rutas)

    modelo_base = AutoModelForSequenceClassification.from_pretrained(
        args.modelo, num_labels=2,
        id2label={0: "legitimo", 1: "phishing"},
        label2id={"legitimo": 0, "phishing": 1})

    # q_lin / v_lin son las proyecciones Q y V de la atencion en DistilBERT.
    # Para otros modelos (BERT, RoBERTa) se llaman query / value.
    objetivo = args.lora_modulos.split(",") if args.lora_modulos else \
        (["q_lin", "v_lin"] if "distilbert" in args.modelo.lower()
         else ["query", "value"])

    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r,
                        lora_alpha=2 * args.lora_r, lora_dropout=0.1,
                        target_modules=objetivo)
    modelo = get_peft_model(modelo_base, config)
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    log.info("Modulos LoRA: %s | entrenables: %d de %d (%.2f%%)",
             objetivo, entrenables, total, 100 * entrenables / total)

    # LoRA usa lr 50 veces mas alto y mas epocas: las matrices delgadas
    # empiezan en cero y necesitan pasos grandes para llegar a algo util.
    trainer = crear_trainer(
        modelo,
        construir_args(rutas.models / args.nombres["dir_lora"], args,
                       epocas=args.epocas_lora, lr=args.lr_lora),
        tokenizer, ds_train, ds_val, collator)

    t0 = time.time()
    trainer.train()
    tiempo = time.time() - t0
    log.info("LoRA entrenado en %.1f minutos", tiempo / 60)

    salida = trainer.predict(ds_test)
    logits = salida.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, axis=-1)
    prob = softmax_np(logits, axis=-1)[:, 1]

    m = metricas(test_df["etiqueta"], pred, prob)
    m["tiempo_entrenamiento_s"] = round(tiempo, 1)
    m["parametros_entrenados"] = int(entrenables)
    acumular_resultado(rutas, args.nombres["res_lora"], m)

    reporte_clasificacion(test_df["etiqueta"], pred)
    modelo.save_pretrained(str(rutas.models / args.nombres["dir_lora"] / "final"))
    np.save(archivo_corrida(args, rutas, "pred_lora.npy"), np.c_[pred, prob])


# ---------------------------------------------------------------------------
# Etapa 6 - Cuantizacion INT8
# ---------------------------------------------------------------------------

def cargar_modelo_ft(rutas: Rutas, args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    destino = rutas.models / args.nombres["dir_ft"] / "final"
    if not destino.exists():
        raise SystemExit(f"No existe {destino}. Corre primero la etapa 'finetune'.")
    modelo = AutoModelForSequenceClassification.from_pretrained(destino).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(destino)
    return modelo, tokenizer


def predecir_cpu(modelo, tokenizer, textos, max_len: int, batch: int = 16):
    """Inferencia en CPU. Devuelve (predicciones, probabilidades, segundos)."""
    import torch
    modelo.eval()
    preds, probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(textos), batch):
            lote = tokenizer(list(textos[i:i + batch]), truncation=True,
                             max_length=max_len, padding=True, return_tensors="pt")
            logits = modelo(**lote).logits
            p = torch.softmax(logits, dim=-1)
            preds.extend(torch.argmax(logits, dim=-1).tolist())
            probs.extend(p[:, 1].tolist())
    return np.array(preds), np.array(probs), time.time() - t0


def etapa_cuantizacion(args, rutas: Rutas) -> None:
    titulo("Etapa 6/9 - Cuantizacion dinamica INT8 (CPU)")

    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score, recall_score

    _, _, test_df = cargar_particiones(rutas)
    modelo_fp32, tokenizer = cargar_modelo_ft(rutas, args)

    # quantize_dynamic se movio a torch.ao.quantization en versiones recientes.
    # La API emite avisos de obsolescencia que aqui solo ensucian el registro.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            from torch.quantization import quantize_dynamic  # type: ignore
        modelo_int8 = quantize_dynamic(modelo_fp32, {nn.Linear}, dtype=torch.qint8)

    def tamano_mb(modelo) -> float:
        tmp = rutas.models / "tmp_size.pt"
        torch.save(modelo.state_dict(), tmp)
        mb = tmp.stat().st_size / 1e6
        tmp.unlink()
        return mb

    # Submuestra para que la medicion en CPU no tarde demasiado
    n = min(args.muestra_cpu, len(test_df))
    sub = test_df.sample(n, random_state=args.seed).reset_index(drop=True)
    log.info("Midiendo sobre %d correos en CPU...", n)

    filas = []
    for nombre, m in (("FP32 (original)", modelo_fp32), ("INT8 (cuantizado)", modelo_int8)):
        p, _, t = predecir_cpu(m, tokenizer, sub["texto"].values, args.max_len)
        filas.append({
            "modelo": nombre,
            "tamano_MB": round(tamano_mb(m), 1),
            "tiempo_total_s": round(t, 1),
            "ms_por_correo": round(1000 * t / len(sub), 1),
            "f1": round(float(f1_score(sub["etiqueta"], p, zero_division=0)), 4),
            "recall": round(float(recall_score(sub["etiqueta"], p, zero_division=0)), 4),
        })
        log.info("%-18s %s", nombre, filas[-1])

    tabla = pd.DataFrame(filas)
    tabla.to_csv(archivo_corrida(args, rutas, "cuantizacion.csv"), index=False)
    log.info("\n%s", tabla.to_string(index=False))


# ---------------------------------------------------------------------------
# Etapa 7 - Generalizacion al espanol
# ---------------------------------------------------------------------------

def plantilla_espanol(ruta: Path) -> None:
    if ruta.exists():
        return
    pd.DataFrame({"texto": ["" for _ in range(100)],
                  "etiqueta": [1] * 50 + [0] * 50}).to_csv(ruta, index=False)
    log.info("Creada la plantilla %s", ruta)
    log.info("Llena la columna 'texto': las primeras 50 filas = phishing, "
             "las siguientes 50 = legitimos. No copies correos reales de nadie "
             "ni incluyas datos personales.")


def leer_espanol(rutas: Rutas) -> pd.DataFrame:
    ruta = rutas.datos / "es_test.csv"
    plantilla_espanol(ruta)
    df = pd.read_csv(ruta).dropna(subset=["texto"])
    df["texto"] = df["texto"].astype(str)
    return df[df["texto"].str.strip().str.len() > 20].reset_index(drop=True)


def etapa_espanol(args, rutas: Rutas) -> None:
    titulo("Etapa 7/9 - Generalizacion a correos en espanol")

    es_df = leer_espanol(rutas)
    if len(es_df) < 20:
        log.warning("Solo hay %d correos llenados en data/es_test.csv. "
                    "Completa el archivo y vuelve a correr esta etapa.", len(es_df))
        return

    modelo, tokenizer = cargar_modelo_ft(rutas, args)
    log.info("%d correos en espanol | phishing: %.0f%%",
             len(es_df), 100 * es_df["etiqueta"].mean())

    pred, prob, _ = predecir_cpu(modelo, tokenizer, es_df["texto"].values, args.max_len)
    m = metricas(es_df["etiqueta"], pred, prob)
    m["n_correos"] = int(len(es_df))
    acumular_resultado(rutas, args.nombres["res_es"], m)

    log.info("Modelo entrenado en ingles, evaluado en espanol:")
    reporte_clasificacion(es_df["etiqueta"], pred)
    graficar_confusion(es_df["etiqueta"], pred,
                       "Modelo sobre correos en espanol",
                       rutas.figs / "05_cm_espanol.png", args.mostrar)

    diagnosticar_espanol(args, rutas, es_df, prob, m)


def diagnosticar_espanol(args, rutas: Rutas, es_df, prob, m: dict) -> None:
    """Separa dos causas distintas de un mal resultado en espanol.

    Si el modelo no entendiera nada, el ROC-AUC seria 0.5: no sabria ordenar
    los correos. Un ROC-AUC alto con accuracy baja significa otra cosa, que el
    orden es bueno pero el umbral de 0.5 quedo en el lugar equivocado porque
    las probabilidades se corrieron al entrar texto fuera de distribucion.
    Distinguirlas cambia por completo la conclusion del reporte.
    """
    from sklearn.metrics import f1_score, accuracy_score

    y = es_df["etiqueta"].values
    log.info("--- Diagnostico ---")
    log.info("Probabilidad media  legitimos: %.3f | phishing: %.3f",
             prob[y == 0].mean(), prob[y == 1].mean())
    log.info("Predichos como phishing con umbral 0.5: %d de %d",
             int((prob > 0.5).sum()), len(prob))
    log.info("ROC-AUC: %.4f  (0.5 seria azar; mide el orden, no el umbral)",
             m.get("roc_auc", float("nan")))

    # Umbral que maximiza F1-macro sobre estos mismos correos. Es una cota
    # optimista: se elige viendo las etiquetas, asi que no es una estimacion
    # honesta de despliegue, solo separa "no entiende" de "no calibra".
    rejilla = np.arange(0.05, 0.999, 0.005)
    mejor = max(rejilla, key=lambda t: f1_score(y, (prob > t).astype(int),
                                                average="macro", zero_division=0))
    pred_cal = (prob > mejor).astype(int)
    m_cal = metricas(y, pred_cal, prob)
    m["umbral_calibrado"] = round(float(mejor), 3)
    m["f1_macro_calibrado"] = round(float(m_cal["f1_macro"]), 4)
    m["accuracy_calibrado"] = round(float(m_cal["accuracy"]), 4)
    acumular_resultado(rutas, args.nombres["res_es"], m)

    log.info("Umbral 0.500 -> accuracy %.4f | F1-macro %.4f",
             accuracy_score(y, (prob > 0.5).astype(int)),
             f1_score(y, (prob > 0.5).astype(int), average="macro", zero_division=0))
    log.info("Umbral %.3f -> accuracy %.4f | F1-macro %.4f  (cota optimista)",
             mejor, m_cal["accuracy"], m_cal["f1_macro"])

    # Fertilidad del tokenizador: cuantos subtokens gasta por palabra. Un
    # numero alto significa que el vocabulario no cubre el idioma y el modelo
    # ve fragmentos sin significado en vez de palabras.
    _, tokenizer = cargar_modelo_ft(rutas, args)
    subtokens = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"])
                    for t in es_df["texto"])
    palabras = sum(len(t.split()) for t in es_df["texto"])
    log.info("Fertilidad del tokenizador en espanol: %.2f subtokens por palabra",
             subtokens / max(palabras, 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(prob[y == 0], bins=bins, alpha=.65, label="legitimos", color="#4C72B0")
    ax.hist(prob[y == 1], bins=bins, alpha=.65, label="phishing", color="#DD8452")
    ax.axvline(0.5, color="black", linestyle="--", label="umbral 0.5")
    ax.axvline(mejor, color="crimson", linestyle=":", label=f"umbral calibrado {mejor:.3f}")
    ax.set_xlabel("Probabilidad de phishing que asigna el modelo")
    ax.set_ylabel("Cantidad de correos")
    ax.set_title("Distribucion de probabilidades en espanol")
    ax.legend()
    fig.tight_layout()
    cerrar_figura(fig, rutas.figs / "07_probs_espanol.png", args.mostrar)


def etapa_beto(args, rutas: Rutas) -> None:
    titulo("Etapa 8/9 - BETO entrenado con los correos en espanol (opcional)")

    from sklearn.model_selection import train_test_split
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding)

    es_df = leer_espanol(rutas)
    if len(es_df) < 60:
        log.warning("Se necesitan al menos 60 correos en espanol (hay %d). "
                    "Etapa BETO omitida.", len(es_df))
        return

    es_tr, es_te = train_test_split(es_df, test_size=0.5, stratify=es_df["etiqueta"],
                                    random_state=args.seed)
    tok = AutoTokenizer.from_pretrained(MODELO_ES)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_ES, num_labels=2)

    ds_tr = tokenizar(es_tr, tok, args.max_len)
    ds_te = tokenizar(es_te, tok, args.max_len)
    collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = crear_trainer(
        modelo, construir_args(rutas.models / "beto", args, epocas=8, lr=2e-5, batch=8),
        tok, ds_tr, ds_te, collator)
    trainer.train()

    salida = trainer.predict(ds_te)
    logits = salida.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred = np.argmax(logits, axis=-1)
    prob = softmax_np(logits, axis=-1)[:, 1]

    m = metricas(es_te["etiqueta"], pred, prob)
    m["n_entrenamiento"] = int(len(es_tr))
    acumular_resultado(rutas, "beto_espanol", m)
    log.info("Recuerda anotar en el reporte que %d ejemplos de entrenamiento son "
             "muy pocos para conclusiones fuertes.", len(es_tr))
    reporte_clasificacion(es_te["etiqueta"], pred)


# ---------------------------------------------------------------------------
# Etapa 9 - Tabla comparativa y registro del entorno
# ---------------------------------------------------------------------------

def etapa_reporte(args, rutas: Rutas) -> None:
    titulo("Etapa 9/9 - Tabla comparativa final")

    import torch
    resultados = cargar_json(rutas.results / "resultados.json")
    if not resultados:
        log.warning("No hay resultados acumulados todavia.")
        return

    tabla = pd.DataFrame(resultados).T
    columnas = ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc",
                "pr_auc", "parametros_entrenados", "tiempo_entrenamiento_s"]
    tabla = tabla[[c for c in columnas if c in tabla.columns]]
    for c in tabla.columns:
        if c == "parametros_entrenados":
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce").astype("Int64")
        else:
            tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(4)
    tabla.to_csv(rutas.results / "comparativa_modelos.csv")
    log.info("\n%s", tabla.to_string())

    # Grafica de barras: F1 y recall de los modelos comparables
    etiquetas = {"baseline_tfidf": "TF-IDF +\nReg. Logistica",
                 "distilbert_finetune": "DistilBERT\nfine-tuning",
                 "distilbert_lora": "DistilBERT\nLoRA"}
    comparables = [k for k in etiquetas if k in resultados]
    if comparables:
        x = np.arange(len(comparables))
        ancho = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(x - ancho / 2, [resultados[k]["f1"] for k in comparables], ancho,
               label="F1 (phishing)", color="#4C72B0")
        ax.bar(x + ancho / 2, [resultados[k]["recall"] for k in comparables], ancho,
               label="Recall (phishing)", color="#DD8452")
        ax.set_xticks(x, [etiquetas[k] for k in comparables])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Valor")
        ax.set_title("Comparacion contra el baseline")
        ax.legend()
        ax.grid(axis="y", alpha=.3)
        for i, k in enumerate(comparables):
            ax.text(i - ancho / 2, resultados[k]["f1"] + .01,
                    f"{resultados[k]['f1']:.3f}", ha="center", fontsize=9)
            ax.text(i + ancho / 2, resultados[k]["recall"] + .01,
                    f"{resultados[k]['recall']:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        cerrar_figura(fig, rutas.figs / "06_comparativa.png", args.mostrar)

    # Registro del entorno: requisito de reproducibilidad de la guia
    import transformers, sklearn, datasets, peft
    entorno = {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "scikit-learn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "fp16": usar_fp16(),
        "semilla": args.seed,
        "max_length": args.max_len,
        "modelo_base": args.modelo,
    }
    guardar_json(entorno, rutas.results / "entorno.json")
    for k, v in entorno.items():
        log.info("%-16s: %s", k, v)


# ---------------------------------------------------------------------------
# Etapa 10 - Demostracion
# ---------------------------------------------------------------------------

EJEMPLOS = [
    "Dear customer, your account has been temporarily suspended. "
    "Please verify your identity within 24 hours at "
    "http://secure-verify-account.com/login or your access will be "
    "permanently terminated.",

    "Hi team, attaching the slides for tomorrow's review meeting. "
    "Let me know if you want anything changed before 5pm. Thanks.",
]


def etapa_demo(args, rutas: Rutas) -> None:
    titulo("Demostracion - clasificacion de texto nuevo")

    import torch
    modelo, tokenizer = cargar_modelo_ft(rutas, args)
    modelo.eval()

    textos = list(args.texto) if args.texto else list(EJEMPLOS)
    if args.archivo_texto:
        textos.extend(l.strip() for l in Path(args.archivo_texto).read_text(
            encoding="utf-8").splitlines() if l.strip())

    for texto in textos:
        entrada = tokenizer(texto, truncation=True, max_length=args.max_len,
                            return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(modelo(**entrada).logits, dim=-1)[0]
        etiqueta = "PHISHING" if p[1] > 0.5 else "LEGITIMO"
        log.info("[%s]  probabilidad de phishing = %.3f", etiqueta, float(p[1]))
        log.info("   %s...", texto[:110])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline de deteccion de phishing con Transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Etapas disponibles: " + ", ".join(ETAPAS))

    p.add_argument("--etapas", default="todas",
                   help="Etapas a ejecutar, separadas por comas. "
                        "'todas' corre el pipeline completo sin BETO.")
    p.add_argument("--salida", default=".", type=Path,
                   help="Directorio donde se escriben data/, models/, results/ y figures/.")
    p.add_argument("--csv", type=Path,
                   help="Corpus local en CSV en vez de descargarlo de Hugging Face.")
    p.add_argument("--forzar", action="store_true",
                   help="Rehace etapas aunque ya existan sus resultados en disco.")

    g = p.add_argument_group("modelo y entrenamiento")
    g.add_argument("--modelo", default=MODELO_EN, help="Modelo base de Hugging Face.")
    g.add_argument("--etiqueta", default="",
                   help="Nombre corto de la corrida. Determina la carpeta en models/, "
                        "las claves en resultados.json y la subcarpeta de figures/, "
                        "para que dos modelos base no se pisen entre si.")
    g.add_argument("--max-len", type=int, default=256, help="Tokens por correo.")
    g.add_argument("--epocas", type=int, default=3, help="Epocas del fine-tuning completo.")
    g.add_argument("--batch", type=int, default=16, help="Tamano de lote de entrenamiento.")
    g.add_argument("--epocas-lora", type=int, default=5)
    g.add_argument("--lr-lora", type=float, default=1e-3)
    g.add_argument("--lora-r", type=int, default=8, help="Rango de las matrices LoRA.")
    g.add_argument("--lora-modulos", default="",
                   help="Modulos objetivo de LoRA separados por comas "
                        "(por defecto q_lin,v_lin en DistilBERT).")

    g2 = p.add_argument_group("otros")
    g2.add_argument("--muestra", type=int, default=0,
                    help="Usa solo N correos del corpus (pruebas rapidas).")
    g2.add_argument("--muestra-cpu", type=int, default=1000,
                    help="Correos usados para medir latencia en la etapa de cuantizacion.")
    g2.add_argument("--mostrar", action="store_true",
                    help="Abre cada figura en una ventana ademas de guardarla. "
                         "El pipeline se detiene hasta que cierras cada una.")
    g2.add_argument("--seed", type=int, default=42)
    g2.add_argument("--texto", action="append",
                    help="Texto a clasificar en la etapa 'demo' (repetible).")
    g2.add_argument("--archivo-texto", type=Path,
                    help="Archivo con un correo por linea para la etapa 'demo'.")
    return p


def resolver_etapas(valor: str) -> list[str]:
    if valor.strip().lower() in ("todas", "all", ""):
        return ETAPAS_POR_DEFECTO
    pedidas = [e.strip().lower() for e in valor.split(",") if e.strip()]
    desconocidas = [e for e in pedidas if e not in ETAPAS]
    if desconocidas:
        raise SystemExit(f"Etapas desconocidas: {desconocidas}. Validas: {ETAPAS}")
    # Se respeta el orden canonico, no el orden en que las escribio el usuario
    return [e for e in ETAPAS if e in pedidas]


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)
    rutas = Rutas(Path(args.salida).resolve())
    rutas.crear()
    configurar_log(rutas.results / "pipeline.log")

    etapas = resolver_etapas(args.etapas)

    # Cada modelo base escribe en su propia carpeta y con sus propias claves
    args.nombres = nombres_corrida(args)
    rutas.figs = (rutas.figures if args.nombres["por_defecto"]
                  else rutas.figures / args.nombres["etiqueta"])
    rutas.figs.mkdir(parents=True, exist_ok=True)

    # El backend por defecto es Agg (sin ventanas) para que el script corra en
    # servidores sin pantalla. Con --mostrar se cambia a uno interactivo.
    if args.mostrar:
        try:
            plt.switch_backend("TkAgg")
        except Exception as e:
            log.warning("No se pudo abrir un backend grafico (%s). Las figuras "
                        "se guardan en figures/ de todos modos.", e)
            args.mostrar = False

    import torch
    fijar_semillas(args.seed)

    titulo("Deteccion de phishing con Transformers")
    log.info("Directorio de trabajo : %s", rutas.base)
    log.info("Etapas a ejecutar     : %s", ", ".join(etapas))
    log.info("Modelo base           : %s", args.modelo)
    log.info("Etiqueta de la corrida: %s", args.nombres["etiqueta"])
    log.info("PyTorch               : %s", torch.__version__)
    if torch.cuda.is_available():
        log.info("GPU                   : %s (fp16=%s)",
                 torch.cuda.get_device_name(0), usar_fp16())
    else:
        log.info("GPU                   : no disponible; el entrenamiento sera "
                 "mucho mas lento en CPU")

    funciones = {
        "datos": etapa_datos,
        "baseline": etapa_baseline,
        "finetune": etapa_finetune,
        "evaluacion": etapa_evaluacion,
        "lora": etapa_lora,
        "cuantizacion": etapa_cuantizacion,
        "espanol": etapa_espanol,
        "beto": etapa_beto,
        "reporte": etapa_reporte,
        "demo": etapa_demo,
    }

    t_inicio = time.time()
    for etapa in etapas:
        funciones[etapa](args, rutas)

    titulo(f"Listo en {(time.time() - t_inicio) / 60:.1f} minutos")
    log.info("Resultados : %s", rutas.results)
    log.info("Figuras    : %s", rutas.figures)
    log.info("Modelos    : %s", rutas.models)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        sys.exit(130)
