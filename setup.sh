#!/usr/bin/env bash
# Crea el entorno virtual e instala las dependencias.
#
#   ./setup.sh            detecta la GPU y elige la rueda de torch adecuada
#   ./setup.sh --cpu      fuerza la version de solo CPU
#
# Usa uv si esta disponible (mucho mas rapido); si no, el venv estandar.
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"
PY_DESEADO="3.12"
FORZAR_CPU=0
[[ "${1:-}" == "--cpu" ]] && FORZAR_CPU=1

# --- Elegir el indice de torch --------------------------------------------
INDICE="https://download.pytorch.org/whl/cpu"
if [[ $FORZAR_CPU -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
    MAYOR=${CAP%%.*}
    if [[ -n "$MAYOR" && "$MAYOR" -le 6 ]]; then
        # Pascal o anterior: las ruedas de PyPI ya no traen codigo para sm_6x
        INDICE="https://download.pytorch.org/whl/cu126"
        echo ">> GPU con capacidad de computo $CAP -> torch CUDA 12.6"
    else
        INDICE=""   # rueda por defecto de PyPI, con la CUDA mas reciente
        echo ">> GPU con capacidad de computo ${CAP:-desconocida} -> torch por defecto (PyPI)"
    fi
else
    echo ">> Sin GPU detectada -> torch de solo CPU"
fi

# --- Crear el entorno ------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo ">> Creando $VENV con uv (Python $PY_DESEADO)"
    uv venv --python "$PY_DESEADO" "$VENV"
    INSTALAR=(uv pip install --python "$VENV/bin/python")
else
    echo ">> Creando $VENV con python3 -m venv"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    INSTALAR=("$VENV/bin/python" -m pip install)
fi

# --- Instalar --------------------------------------------------------------
if [[ -n "$INDICE" ]]; then
    "${INSTALAR[@]}" torch --index-url "$INDICE"
fi
"${INSTALAR[@]}" -r requirements.txt

echo
echo "Listo. Para usarlo:"
echo "    source $VENV/bin/activate"
echo "    ./pipeline_phishing.py --help"
