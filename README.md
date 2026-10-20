# Sistema de control por IA local

Aplicación de escritorio en Python con CustomTkinter y Ollama. Incluye herramientas
de robótica, redes neuronales, visión por computadora, lógica difusa, PLN y
optimización.

## Requisitos

- Python 3.10 o posterior
- Ollama instalado y en ejecución
- El modelo local `qwen3:1.7b`

## Instalación

Desde esta carpeta, instala las dependencias y descarga el modelo:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

## Ejecución

```powershell
python "tarea ia.py"
```

También se incluye el lanzador `tarea ia` para configuraciones de ejecución
anteriores que todavía usen ese nombre sin extensión.

En el chat se pueden escribir instrucciones en lenguaje natural. Para analizar
una imagen, usa **SUBIR IMAGEN** o copia una imagen y pulsa **Ctrl+Shift+V**.
