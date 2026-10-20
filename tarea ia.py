"""
=============================================================================
SISTEMA DE CONTROL TOTAL POR IA LOCAL (OLLAMA) - INTERFAZ MODERNA 2026
=============================================================================
La IA controla los módulos del sistema mediante Tool Calling.
No requiere API Key ni conexión a internet.
=============================================================================
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import math
import random
import re
import time
from collections import Counter
import heapq
import threading
import ollama
import json
from PIL import Image, ImageGrab, UnidentifiedImageError

# =============================================================================
# CONFIGURACIÓN DE APARIENCIA MODERNA
# =============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
OLLAMA_MODEL = "qwen3:1.7b"

# =============================================================================
# LÓGICA DE IA (MÓDULOS MATEMÁTICOS)
# =============================================================================

class Utils:
    @staticmethod
    def distancia_manhattan(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    @staticmethod
    def sigmoide(x):
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))
    @staticmethod
    def derivada_sigmoide(x): return x * (1.0 - x)

class EntornoRobotica:
    def __init__(self, ancho, alto, obstaculos):
        self.ancho, self.alto, self.obstaculos = ancho, alto, set(obstaculos)
    def es_transitable(self, pos):
        x, y = pos
        return 0 <= x < self.ancho and 0 <= y < self.alto and pos not in self.obstaculos
    def vecinos(self, pos):
        x, y = pos
        return [(x+dx, y+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)] if self.es_transitable((x+dx, y+dy))]

class AlgoritmoAStar:
    def __init__(self, entorno): self.entorno = entorno
    def buscar(self, inicio, fin):
        abiertos = [(0, inicio)]
        came_from, g_score = {}, {inicio: 0}
        while abiertos:
            _, actual = heapq.heappop(abiertos)
            if actual == fin:
                camino = []
                while actual in came_from:
                    camino.append(actual); actual = came_from[actual]
                camino.append(inicio)
                return camino[::-1]
            for vecino in self.entorno.vecinos(actual):
                tentative_g = g_score[actual] + 1
                if tentative_g < g_score.get(vecino, float('inf')):
                    came_from[vecino] = actual
                    g_score[vecino] = tentative_g
                    f = tentative_g + Utils.distancia_manhattan(vecino, fin)
                    heapq.heappush(abiertos, (f, vecino))
        return None

class Neurona:
    def __init__(self, num_entradas):
        self.pesos = [random.uniform(-1, 1) for _ in range(num_entradas)]
        self.sesgo = random.uniform(-1, 1)
        self.salida, self.entradas = 0.0, []
    def activar(self, entradas):
        self.entradas = entradas
        self.salida = Utils.sigmoide(sum(e * p for e, p in zip(entradas, self.pesos)) + self.sesgo)
        return self.salida
    def ajustar(self, delta, tasa):
        for i in range(len(self.pesos)): self.pesos[i] += tasa * delta * self.entradas[i]
        self.sesgo += tasa * delta

class RedNeuronalMLP:
    def __init__(self, capas):
        self.capas = [[Neurona(capas[i-1]) for j in range(capas[i])] for i in range(1, len(capas))]
    def forward(self, entradas):
        act = entradas
        for capa in self.capas: act = [n.activar(act) for n in capa]
        return act
    def backward(self, entradas, objetivos, tasa):
        activaciones = [entradas]; act = entradas
        for capa in self.capas:
            act = [n.activar(act) for n in capa]; activaciones.append(act)
        deltas = []; capa_salida = self.capas[-1]; delta_salida = []
        for i, n in enumerate(capa_salida):
            delta = (objetivos[i] - n.salida) * Utils.derivada_sigmoide(n.salida)
            delta_salida.append(delta); n.ajustar(delta, tasa)
        deltas.insert(0, delta_salida)
        for l in range(len(self.capas) - 2, -1, -1):
            capa, capa_sig = self.capas[l], self.capas[l+1]
            delta_capa = []
            for i, n in enumerate(capa):
                error = sum(deltas[0][j] * capa_sig[j].pesos[i] for j in range(len(capa_sig)))
                delta = error * Utils.derivada_sigmoide(n.salida)
                delta_capa.append(delta); n.ajustar(delta, tasa)
            deltas.insert(0, delta_capa)
    def entrenar(self, X, Y, epocas, tasa):
        error_final = 0
        for _ in range(epocas):
            error_final = 0
            for x, y in zip(X, Y):
                pred = self.forward(x)
                error_final += sum((y[i] - pred[i])**2 for i in range(len(y)))
                self.backward(x, y, tasa)
        return error_final / len(X)

class FiltrosImagen:
    @staticmethod
    def convolucion(matriz, kernel):
        k_alto, k_ancho = len(kernel), len(kernel[0])
        pad_h, pad_w = k_alto // 2, k_ancho // 2
        alto, ancho = len(matriz), len(matriz[0])
        pad_matriz = [[0]*(ancho + 2*pad_w) for _ in range(alto + 2*pad_h)]
        for i in range(alto):
            for j in range(ancho): pad_matriz[i+pad_h][j+pad_w] = matriz[i][j]
        resultado = [[0]*ancho for _ in range(alto)]
        for i in range(alto):
            for j in range(ancho):
                suma = 0
                for ki in range(k_alto):
                    for kj in range(k_ancho): suma += pad_matriz[i+ki][j+kj] * kernel[ki][kj]
                resultado[i][j] = suma
        return resultado
    @staticmethod
    def sobel(matriz):
        kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        gx, gy = FiltrosImagen.convolucion(matriz, kx), FiltrosImagen.convolucion(matriz, ky)
        res = [[0]*len(matriz[0]) for _ in range(len(matriz))]
        for i in range(len(matriz)):
            for j in range(len(matriz[0])):
                res[i][j] = max(0, min(255, int(math.sqrt(gx[i][j]**2 + gy[i][j]**2))))
        return res

class ConjuntoDifuso:
    def __init__(self, a, b, c, d): self.a, self.b, self.c, self.d = a, b, c, d
    def pertenencia(self, x):
        if self.a == self.b and x <= self.b: return 1.0
        if self.c == self.d and x >= self.c: return 1.0
        if x <= self.a or x >= self.d: return 0.0
        if self.b <= x <= self.c: return 1.0
        if self.a < x < self.b: return (x - self.a) / (self.b - self.a)
        if self.c < x < self.d: return (self.d - x) / (self.d - self.c)
        return 0.0

class ControladorDifuso:
    def __init__(self):
        self.frio = ConjuntoDifuso(-10, -10, 15, 25)
        self.templado = ConjuntoDifuso(18, 25, 30, 38)
        self.caliente = ConjuntoDifuso(32, 42, 60, 60)
        self.baja = ConjuntoDifuso(0, 0, 20, 45)
        self.media = ConjuntoDifuso(30, 48, 58, 78)
        self.alta = ConjuntoDifuso(60, 80, 100, 100)

    def calcular(self, temp):
        p_frio = self.frio.pertenencia(temp)
        p_templado = self.templado.pertenencia(temp)
        p_caliente = self.caliente.pertenencia(temp)
        activaciones = (
            (p_frio, 0.0),
            (p_templado, 50.0),
            (p_caliente, 100.0),
        )
        peso_total = sum(peso for peso, _ in activaciones)
        if peso_total == 0:
            return 0.0 if temp < 18 else 100.0
        velocidad = sum(peso * salida for peso, salida in activaciones) / peso_total
        return max(0.0, min(100.0, velocidad))

class ClasificadorPLN:
    def __init__(self):
        self.vocab, self.c_spam, self.c_ham = set(), Counter(), Counter()
        self.t_spam, self.t_ham = 0, 0
    def tokenizar(self, texto): return re.findall(r'\b\w+\b', texto.lower())
    def entrenar(self, spam, ham):
        for txt in spam:
            self.c_spam.update(self.tokenizar(txt)); self.t_spam += 1
        for txt in ham:
            self.c_ham.update(self.tokenizar(txt)); self.t_ham += 1
        self.vocab = set(self.c_spam.keys()).union(set(self.c_ham.keys()))
    def clasificar(self, texto):
        tokens = self.tokenizar(texto)
        p_spam = math.log(self.t_spam / (self.t_spam + self.t_ham))
        p_ham = math.log(self.t_ham / (self.t_spam + self.t_ham))
        v_size, t_pal_spam, t_pal_ham = len(self.vocab), sum(self.c_spam.values()), sum(self.c_ham.values())
        for t in tokens:
            if t in self.vocab:
                p_spam += math.log((self.c_spam[t] + 1) / (t_pal_spam + v_size))
                p_ham += math.log((self.c_ham[t] + 1) / (t_pal_ham + v_size))
        return "SPAM" if p_spam > p_ham else "HAM (No Spam)"

class MotorExperto:
    def __init__(self):
        self.reglas = [
            ({"fiebre", "tos"}, "gripe"),
            ({"fiebre", "erupcion"}, "sarampion"),
            ({"gripe", "dolor_cabeza"}, "reposo"),
            ({"sarampion", "fiebre_alta"}, "hospital")
        ]
    def inferir(self, hechos_iniciales):
        hechos, nuevos, pasos = set(hechos_iniciales), True, []
        while nuevos:
            nuevos = False
            for cond, cons in self.reglas:
                if cond.issubset(hechos) and cons not in hechos:
                    hechos.add(cons); pasos.append(f"SI {cond} -> {cons}"); nuevos = True
        return hechos, pasos

class Optimizador:
    @staticmethod
    def funcion(x, y): return 20 + x**2 + y**2 - 10 * (math.cos(2*math.pi*x) + math.cos(2*math.pi*y))
    @staticmethod
    def genetico():
        pob = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(50)]
        for _ in range(100):
            pob.sort(key=lambda ind: Optimizador.funcion(ind[0], ind[1]))
            mejores = pob[:10]
            nueva = mejores[:]
            while len(nueva) < 50:
                p1, p2 = random.choice(mejores), random.choice(mejores)
                nueva.append(((p1[0]+p2[0])/2 + random.gauss(0, 0.5), (p1[1]+p2[1])/2 + random.gauss(0, 0.5)))
            pob = nueva
        return min(pob, key=lambda ind: Optimizador.funcion(ind[0], ind[1]))
    @staticmethod
    def annealing():
        x, y = random.uniform(-5, 5), random.uniform(-5, 5)
        mejor_x, mejor_y, mejor_v = x, y, Optimizador.funcion(x, y)
        temp = 100.0
        while temp > 0.01:
            nx, ny = x + random.uniform(-1, 1), y + random.uniform(-1, 1)
            nv = Optimizador.funcion(nx, ny)
            if nv < mejor_v or random.random() < math.exp((mejor_v - nv) / temp):
                x, y = nx, ny
                if nv < mejor_v: mejor_v, mejor_x, mejor_y = nv, x, y
            temp *= 0.95
        return mejor_x, mejor_y

# =============================================================================
# HERRAMIENTAS (FUNCIONES QUE LA IA PUEDE EJECUTAR)
# =============================================================================

def ejecutar_robotica() -> str:
    """Simula un robot buscando una ruta en un mapa con obstáculos."""
    obstaculos = [(1,1), (2,1), (3,1), (4,1), (4,2), (4,3), (1,4), (2,4), (3,4)]
    entorno = EntornoRobotica(8, 8, obstaculos)
    camino = AlgoritmoAStar(entorno).buscar((0, 0), (7, 7))
    if camino:
        mapa = []
        for y in range(entorno.alto):
            fila = []
            for x in range(entorno.ancho):
                posicion = (x, y)
                fila.append("R" if posicion == camino[0] else
                            "F" if posicion == camino[-1] else
                            "#" if posicion in obstaculos else
                            "*" if posicion in camino else ".")
            mapa.append(" ".join(fila))
        return f"Ruta encontrada con {len(camino) - 1} movimientos.\n" + "\n".join(mapa)
    return "No se encontró ruta."

def ejecutar_red_neuronal() -> str:
    """Entrena una red neuronal para aprender la compuerta lógica XOR."""
    mlp = RedNeuronalMLP([2, 3, 1])
    X, Y = [[0,0], [0,1], [1,0], [1,1]], [[0], [1], [1], [0]]
    error = mlp.entrenar(X, Y, epocas=3000, tasa=0.5)
    predicciones = [
        f"{entrada} -> {mlp.forward(entrada)[0]:.3f}"
        for entrada in X
    ]
    return f"Red neuronal entrenada. Error final: {error:.6f}\n" + "\n".join(predicciones)

def ejecutar_vision() -> str:
    """Aplica el filtro Sobel para detectar bordes en una imagen simulada."""
    img = [[0]*6 for _ in range(6)]
    for i in range(2, 4):
        for j in range(2, 4): img[i][j] = 255
    bordes = FiltrosImagen.sobel(img)
    matriz = "\n".join(" ".join(f"{valor:3d}" for valor in fila) for fila in bordes)
    return f"Bordes detectados:\n{matriz}"

def ejecutar_logica_difusa(temperatura: float) -> str:
    """Calcula la velocidad de un ventilador basado en la temperatura usando lógica difusa."""
    vel = ControladorDifuso().calcular(temperatura)
    return f"A {temperatura}°C, la velocidad del ventilador es {vel:.2f}%."

def ejecutar_pln(mensaje: str) -> str:
    """Clasifica un mensaje de texto como SPAM o HAM (No Spam)."""
    modelo = ClasificadorPLN()
    modelo.entrenar(["Gana dinero rápido", "Oferta increíble"], ["Hola cómo estás", "Reunión mañana"])
    res = modelo.clasificar(mensaje)
    return f"El mensaje '{mensaje}' ha sido clasificado como: {res}."

def ejecutar_sistema_experto(sintomas: str) -> str:
    """Diagnostica posibles enfermedades basado en una lista de síntomas (separados por coma)."""
    hechos = [s.strip().lower() for s in sintomas.split(",")]
    res, pasos = MotorExperto().inferir(hechos)
    diag = [h for h in res if h in ["gripe", "sarampion", "reposo", "hospital"]]
    detalle = "\n".join(f"- {paso}" for paso in pasos) or "- No se activaron reglas."
    return f"Análisis de {hechos}:\n{detalle}\nConclusión: {diag if diag else 'Ninguna regla concluyente'}."

def ejecutar_algoritmo_genetico() -> str:
    """Optimiza la función de Rastrigin usando un Algoritmo Genético."""
    x, y = Optimizador.genetico()
    return f"Algoritmo Genético encontró la solución: x={x:.4f}, y={y:.4f}."

def ejecutar_simulated_annealing() -> str:
    """Optimiza la función de Rastrigin usando Simulated Annealing."""
    x, y = Optimizador.annealing()
    return f"Simulated Annealing encontró la solución: x={x:.4f}, y={y:.4f}."

# =============================================================================
# INTERFAZ GRÁFICA (CUSTOMTKINTER)
# =============================================================================

class AppControlIA(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SISTEMA DE CONTROL TOTAL POR IA LOCAL - 2026")
        self.geometry("1200x800")
        self.historial = [{
            "role": "system",
            "content": (
                "Eres el asistente de un programa local. Responde siempre en español, "
                "de forma natural y clara. Interpreta instrucciones abiertas y expresadas "
                "con palabras distintas; no exijas que el usuario conozca nombres de módulos. "
                "Usa las herramientas cuando pidan ejecutar, analizar, calcular o simular "
                "algo compatible, y combina varias si la petición tiene varias partes. "
                "Para conversación general responde directamente. Si falta un dato esencial, "
                "pregunta solo por ese dato. Usa los resultados reales de las herramientas "
                "para responder; no inventes resultados. Analiza internamente y ofrece una "
                "explicación breve, sin revelar razonamiento privado paso a paso."
            ),
        }]
        self.procesando = False
        self.animando_estado = False
        self.cola_resultados = []
        self.animando_resultado = False
        self.animaciones = {}
        self.contexto_imagen = ""
        self.imagen_preview = None
        self.fan_frame = 0
        
        # Panel Izquierdo: Chat de Control
        self.frame_chat = ctk.CTkFrame(self, width=400, corner_radius=0)
        self.frame_chat.pack(side="left", fill="y", padx=0, pady=0)
        
        ctk.CTkLabel(self.frame_chat, text="IA LOCAL (OLLAMA)", font=("Consolas", 20, "bold")).pack(pady=20)
        
        self.chat_area = ctk.CTkTextbox(self.frame_chat, state="disabled", font=("Consolas", 12))
        self.chat_area.pack(fill="both", expand=True, padx=15, pady=10)
        self.image_preview_label = ctk.CTkLabel(self.frame_chat, text="")

        self.estado_ia = ctk.CTkLabel(
            self.frame_chat,
            text="● Lista para conversar",
            anchor="w",
            font=("Consolas", 11),
            text_color="#53d6a0",
        )
        self.estado_ia.pack(fill="x", padx=15, pady=(0, 4))
        self.progreso = ctk.CTkProgressBar(self.frame_chat, mode="indeterminate", height=5)
        self.progreso.pack(fill="x", padx=15, pady=(0, 7))
        self.progreso.set(0)

        self.entry_prompt = ctk.CTkEntry(
            self.frame_chat,
            placeholder_text="Escribe lo que necesitas, con tus propias palabras...",
            font=("Consolas", 12),
        )
        self.entry_prompt.pack(fill="x", padx=15, pady=5)
        self.entry_prompt.bind("<Return>", lambda e: self.enviar_comando())

        self.entry_prompt.bind("<Control-Shift-v>", self.pegar_imagen_portapapeles)
        self.entry_prompt.bind("<Control-Shift-V>", self.pegar_imagen_portapapeles)
        self.frame_imagen = ctk.CTkFrame(self.frame_chat, fg_color="transparent")
        self.frame_imagen.pack(fill="x", padx=15, pady=(2, 4))
        self.btn_subir_imagen = ctk.CTkButton(
            self.frame_imagen,
            text="SUBIR IMAGEN",
            command=self.subir_imagen,
            height=30,
            fg_color="#245c60",
            hover_color="#327b78",
        )
        self.btn_subir_imagen.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_pegar_imagen = ctk.CTkButton(
            self.frame_imagen,
            text="PEGAR (Ctrl+Shift+V)",
            command=self.pegar_imagen_portapapeles,
            height=30,
            fg_color="#245c60",
            hover_color="#327b78",
        )
        self.btn_pegar_imagen.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.btn_enviar = ctk.CTkButton(self.frame_chat, text="ENVIAR ORDEN", command=self.enviar_comando, font=("Consolas", 12, "bold"))
        self.btn_enviar.pack(fill="x", padx=15, pady=(5, 20))
        
        # Panel Derecho: Visualización de Módulos
        self.frame_modulos = ctk.CTkFrame(self, corner_radius=0)
        self.frame_modulos.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        
        self.tabview = ctk.CTkTabview(self.frame_modulos, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.tab_robotica = self.tabview.add("4.1 Robótica")
        self.tab_rn = self.tabview.add("4.2 Redes Neuronales")
        self.tab_vision = self.tabview.add("4.3 Visión")
        self.tab_difusa = self.tabview.add("4.4 Lógica Difusa")
        self.tab_pln = self.tabview.add("4.5 PLN")
        self.tab_experto = self.tabview.add("4.6 Sist. Expertos")
        self.tab_opt = self.tabview.add("Extra: Optimización")
        
        # Consolas de salida para cada módulo
        self.visualizaciones = {}
        self._crear_visualizacion("robotica", self.tab_robotica, "RUTA DEL ROBOT")
        self._crear_visualizacion("red", self.tab_rn, "PROPAGACIÓN NEURONAL")
        self._crear_visualizacion("vision", self.tab_vision, "BARRIDO DE VISIÓN")
        self._crear_visualizacion("difusa", self.tab_difusa, "CONTROL DIFUSO")
        self._crear_visualizacion("pln", self.tab_pln, "ANÁLISIS DEL MENSAJE")
        self._crear_visualizacion("experto", self.tab_experto, "MOTOR DE INFERENCIA")
        self._crear_visualizacion("opt", self.tab_opt, "BÚSQUEDA DE SOLUCIÓN")
        self._crear_control_ventilador()

        self.txt_robotica = ctk.CTkTextbox(self.tab_robotica, font=("Courier", 12))
        self.txt_robotica.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_rn = ctk.CTkTextbox(self.tab_rn, font=("Courier", 12))
        self.txt_rn.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_vision = ctk.CTkTextbox(self.tab_vision, font=("Courier", 12))
        self.txt_vision.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_difusa = ctk.CTkTextbox(self.tab_difusa, font=("Courier", 12))
        self.txt_difusa.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_pln = ctk.CTkTextbox(self.tab_pln, font=("Courier", 12))
        self.txt_pln.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_experto = ctk.CTkTextbox(self.tab_experto, font=("Courier", 12))
        self.txt_experto.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.txt_opt = ctk.CTkTextbox(self.tab_opt, font=("Courier", 12))
        self.txt_opt.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Iniciar chat
        self.log_chat(
            "Sistema listo. Puedes conversar y pedir tareas con tus propias palabras; "
            "también puedes combinar varias solicitudes."
        )
        
        # Verificar conexión con Ollama
        self.verificar_ollama()

    def _crear_control_ventilador(self):
        controles = ctk.CTkFrame(self.tab_difusa, fg_color="transparent")
        controles.pack(fill="x", padx=14, pady=(4, 0))
        self.etiqueta_temperatura = ctk.CTkLabel(
            controles,
            text="Temperatura: 20°C  |  Potencia: 18%",
            font=("Consolas", 11, "bold"),
        )
        self.etiqueta_temperatura.pack(anchor="w")
        self.control_temperatura = ctk.CTkSlider(
            controles,
            from_=-10,
            to=60,
            number_of_steps=140,
            command=self._actualizar_temperatura,
        )
        self.control_temperatura.set(20)
        self.control_temperatura.pack(fill="x", pady=(2, 5))
        self._actualizar_temperatura(20)

    def _actualizar_temperatura(self, valor):
        temperatura = float(valor)
        potencia = ControladorDifuso().calcular(temperatura)
        self.etiqueta_temperatura.configure(
            text=f"Temperatura: {temperatura:.1f}°C  |  Potencia: {potencia:.1f}%"
        )
        canvas = self.visualizaciones.get("difusa")
        if canvas is not None and canvas.winfo_exists():
            self.fan_frame += 1
            canvas.delete("all")
            self._dibujar_ventilador(
                canvas,
                max(canvas.winfo_width(), 400) / 2,
                max(canvas.winfo_height(), 170) / 2,
                self.fan_frame,
                potencia / 100,
                temperatura,
            )

    def subir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen para analizar",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self._iniciar_analisis_imagen(ruta, ruta.rsplit("\\", 1)[-1])

    def pegar_imagen_portapapeles(self, event=None):
        try:
            contenido = ImageGrab.grabclipboard()
        except OSError as error:
            self.log_chat(f"No se pudo leer el portapapeles: {error}")
            return "break" if event else None

        if isinstance(contenido, Image.Image):
            self._iniciar_analisis_imagen(contenido.copy(), "Imagen del portapapeles")
        elif isinstance(contenido, list):
            ruta = next(
                (elemento for elemento in contenido if isinstance(elemento, str)),
                None,
            )
            if ruta:
                self._iniciar_analisis_imagen(ruta, ruta.rsplit("\\", 1)[-1])
            else:
                self.log_chat("El portapapeles no contiene una imagen compatible.")
        else:
            self.log_chat(
                "No hay una imagen en el portapapeles. Copia una imagen y vuelve a pegarla."
            )
        return "break" if event else None

    def _iniciar_analisis_imagen(self, origen, nombre):
        self.estado_ia.configure(text="● Analizando imagen y bordes")
        self.btn_subir_imagen.configure(state="disabled")
        self.btn_pegar_imagen.configure(state="disabled")
        threading.Thread(
            target=self._analizar_imagen_en_segundo_plano,
            args=(origen, nombre),
            daemon=True,
        ).start()

    def _analizar_imagen_en_segundo_plano(self, origen, nombre):
        try:
            if isinstance(origen, Image.Image):
                imagen = origen.copy()
            else:
                with Image.open(origen) as archivo:
                    imagen = archivo.copy()
            imagen = imagen.convert("L")
            ancho_original, alto_original = imagen.size
            if not ancho_original or not alto_original:
                raise ValueError("La imagen está vacía.")

            imagen.thumbnail((160, 160), Image.Resampling.LANCZOS)
            pixeles = list(imagen.getdata())
            matriz = [
                [int(valor) for valor in pixeles[inicio:inicio + imagen.width]]
                for inicio in range(0, len(pixeles), imagen.width)
            ]
            bordes = FiltrosImagen.sobel(matriz)

            vista = imagen.copy()
            vista.thumbnail((20, 20), Image.Resampling.LANCZOS)
            pixeles_vista = list(vista.getdata())
            matriz_vista = [
                [int(valor) for valor in pixeles_vista[inicio:inicio + vista.width]]
                for inicio in range(0, len(pixeles_vista), vista.width)
            ]
            vista_bordes = Image.frombytes(
                "L",
                (imagen.width, imagen.height),
                bytes(valor for fila in bordes for valor in fila),
            )
            vista_bordes = vista_bordes.resize(vista.size, Image.Resampling.LANCZOS)
            pixeles_bordes = list(vista_bordes.getdata())
            matriz_bordes = [
                [int(valor) for valor in pixeles_bordes[inicio:inicio + vista.width]]
                for inicio in range(0, len(pixeles_bordes), vista.width)
            ]

            pixeles_con_borde = sum(valor >= 80 for fila in bordes for valor in fila)
            resumen = (
                f"Imagen: {nombre}. Resolución original: {ancho_original}x{alto_original}. "
                f"Análisis Sobel: {pixeles_con_borde} píxeles de borde detectados "
                f"en la imagen procesada de {imagen.width}x{imagen.height}."
            )
            self.after(
                0,
                self._mostrar_imagen_analizada,
                nombre,
                resumen,
                matriz_vista,
                matriz_bordes,
                vista,
            )
        except (OSError, TypeError, ValueError, UnidentifiedImageError) as error:
            self.after(0, self._error_analisis_imagen, f"No se pudo analizar la imagen: {error}")

    def _error_analisis_imagen(self, mensaje):
        self.log_chat(mensaje)
        self.estado_ia.configure(text="● Lista para conversar")
        self.btn_subir_imagen.configure(state="normal")
        self.btn_pegar_imagen.configure(state="normal")

    def _mostrar_imagen_analizada(
        self,
        nombre,
        resumen,
        matriz_original,
        matriz_bordes,
        imagen_preview,
    ):
        self.contexto_imagen = resumen
        self.log_chat(f"IMAGEN: {resumen}")
        self.tabview.set("4.3 Visión")
        self._animar_visualizacion(
            "vision",
            resumen,
            {"original": matriz_original, "matriz": matriz_bordes},
        )
        self.log_modulo(self.txt_vision, f"--- IMAGEN ANALIZADA: {nombre} ---")
        self.log_modulo(self.txt_vision, resumen)

        imagen_preview.thumbnail((120, 76), Image.Resampling.LANCZOS)
        self.imagen_preview = ctk.CTkImage(
            light_image=imagen_preview,
            dark_image=imagen_preview,
            size=imagen_preview.size,
        )
        self.image_preview_label.configure(
            text=f"Adjunta: {nombre}",
            image=self.imagen_preview,
            compound="left",
            anchor="w",
        )
        self.image_preview_label.pack(fill="x", padx=15, pady=(0, 5))
        self.estado_ia.configure(text="● Imagen analizada; puedes preguntar sobre sus bordes")
        self.btn_subir_imagen.configure(state="normal")
        self.btn_pegar_imagen.configure(state="normal")

    def _crear_visualizacion(self, nombre, pestaña, titulo):
        contenedor = ctk.CTkFrame(pestaña, fg_color="transparent")
        contenedor.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(
            contenedor,
            text=titulo,
            anchor="w",
            font=("Consolas", 11, "bold"),
            text_color="#73e0bd",
        ).pack(fill="x", padx=4, pady=(0, 5))
        canvas = ctk.CTkCanvas(
            contenedor,
            height=190,
            background="#171d22",
            highlightthickness=1,
            highlightbackground="#33434b",
        )
        canvas.pack(fill="x", expand=True)
        self.visualizaciones[nombre] = canvas
        self.after(100, self._dibujar_visual_inicial, nombre)

    def _dibujar_visual_inicial(self, nombre):
        canvas = self.visualizaciones[nombre]
        ancho = max(canvas.winfo_width(), 400)
        alto = 190
        centro_x, centro_y = ancho / 2, alto / 2
        canvas.delete("all")
        if nombre == "robotica":
            self._dibujar_mapa(canvas, ancho, alto, None, 0)
        elif nombre == "red":
            self._dibujar_red(canvas, ancho, alto, 0)
        elif nombre == "vision":
            self._dibujar_vision(canvas, ancho, alto, 0, None)
        elif nombre == "difusa":
            temperatura = float(self.control_temperatura.get())
            potencia = ControladorDifuso().calcular(temperatura)
            self._dibujar_ventilador(
                canvas, centro_x, centro_y, 0, potencia / 100, temperatura
            )
        elif nombre == "pln":
            self._dibujar_pln(canvas, ancho, alto, 0, "", None)
        elif nombre == "experto":
            self._dibujar_experto(canvas, ancho, alto, 0, [], "")
        elif nombre == "opt":
            self._dibujar_optimizacion(canvas, ancho, alto, 0, None)

    def verificar_ollama(self):
        try:
            modelos = ollama.list().models
            nombres = {
                getattr(modelo, "model", None) or getattr(modelo, "name", None)
                for modelo in modelos
            }
            if OLLAMA_MODEL not in nombres:
                self.log_chat(
                    f"Falta el modelo '{OLLAMA_MODEL}'. Instálalo desde una terminal "
                    f"con: ollama pull {OLLAMA_MODEL}",
                    "orange"
                )
                return
            self.log_chat("✅ Conexión con Ollama establecida.", "green")
        except Exception as error:
            self.log_chat(
                f"❌ No se pudo conectar con Ollama: {error}. "
                "Comprueba que Ollama esté abierto o ejecuta 'ollama serve'.",
                "red"
            )

    def log_chat(self, mensaje, color="white"):
        self.chat_area.configure(state="normal")
        self.chat_area.insert("end", mensaje + "\n\n")
        # Nota: CustomTkinter no soporta colores por tag fácilmente, pero el texto es legible.
        self.chat_area.see("end")
        self.chat_area.configure(state="disabled")

    def log_modulo(self, consola, mensaje):
        consola.configure(state="normal")
        consola.insert("end", mensaje + "\n")
        consola.see("end")
        consola.configure(state="disabled")

    def enviar_comando(self):
        if self.procesando:
            return
        prompt = self.entry_prompt.get().strip()
        if not prompt: return
        
        self.entry_prompt.delete(0, "end")
        self.log_chat(f"TÚ: {prompt}")
        mensaje_usuario = prompt
        if self.contexto_imagen:
            mensaje_usuario += (
                "\n\nContexto de la imagen adjunta, obtenido mediante análisis local "
                f"de píxeles y bordes: {self.contexto_imagen}"
            )
        self.historial.append({"role": "user", "content": mensaje_usuario})
        self.procesando = True
        self.btn_enviar.configure(state="disabled", text="PENSANDO...")
        self.estado_ia.configure(text="● Entendiendo tu solicitud")
        self.progreso.start()
        self.animando_estado = True
        self._animar_estado(0)
        
        threading.Thread(target=self.procesar_ia, daemon=True).start()

    def procesar_ia(self):
        herramientas = [
            ejecutar_robotica, ejecutar_red_neuronal, ejecutar_vision,
            ejecutar_logica_difusa, ejecutar_pln, ejecutar_sistema_experto,
            ejecutar_algoritmo_genetico, ejecutar_simulated_annealing
        ]
        funciones = {funcion.__name__: funcion for funcion in herramientas}
        parametros_requeridos = {
            "ejecutar_logica_difusa": ("temperatura",),
            "ejecutar_pln": ("mensaje",),
            "ejecutar_sistema_experto": ("sintomas",),
        }

        try:
            for _ in range(5):
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=self.historial,
                    tools=herramientas,
                    options={"temperature": 0.35},
                    keep_alive="5m",
                )
                mensaje = response.message
                self.historial.append(mensaje)
                llamadas = mensaje.tool_calls or []
                if not llamadas:
                    respuesta = (mensaje.content or "").strip()
                    if respuesta:
                        self.after(0, self.log_chat, f"IA: {respuesta}")
                    else:
                        self.after(
                            0,
                            self.log_chat,
                            "No recibí una respuesta del modelo. Intenta reformular tu solicitud.",
                        )
                    break

                for llamada in llamadas:
                    nombre = llamada.function.name
                    argumentos = llamada.function.arguments or {}
                    self.after(
                        0,
                        self.estado_ia.configure,
                        {"text": f"● Ejecutando {nombre.replace('ejecutar_', '').replace('_', ' ')}"},
                    )

                    funcion = funciones.get(nombre)
                    if funcion is None:
                        resultado = f"Error: la herramienta '{nombre}' no está disponible."
                    elif not isinstance(argumentos, dict):
                        resultado = f"Error: argumentos no válidos para {nombre}."
                    else:
                        faltantes = [
                            clave for clave in parametros_requeridos.get(nombre, ())
                            if clave not in argumentos
                        ]
                        if faltantes:
                            resultado = (
                                f"Falta el dato requerido ({', '.join(faltantes)}). "
                                "Pregunta al usuario por ese dato antes de volver a intentarlo."
                            )
                        else:
                            argumentos = {
                                clave: valor
                                for clave, valor in argumentos.items()
                                if clave in parametros_requeridos.get(nombre, ())
                            }
                            if nombre == "ejecutar_logica_difusa":
                                try:
                                    argumentos["temperatura"] = float(argumentos["temperatura"])
                                except (TypeError, ValueError):
                                    resultado = "La temperatura debe ser un número."
                                    argumentos = None

                            if argumentos is not None:
                                try:
                                    resultado = str(funcion(**argumentos))
                                    self.after(
                                        0,
                                        self.mostrar_resultado_herramienta,
                                        nombre,
                                        resultado,
                                    )
                                except Exception as error:
                                    resultado = f"Error al ejecutar {nombre}: {error}"

                    self.historial.append({
                        "role": "tool",
                        "name": nombre,
                        "content": resultado,
                    })
            else:
                self.after(
                    0,
                    self.log_chat,
                    "Alcancé el límite de pasos para esta solicitud. Puedes pedirme que continúe.",
                )

            if len(self.historial) > 16:
                self.historial = [self.historial[0], *self.historial[-14:]]
        except Exception as e:
            self.after(0, self.log_chat, f"Error del sistema: {e}")
        finally:
            self.after(0, self.finalizar_procesamiento)

    def finalizar_procesamiento(self):
        self.procesando = False
        self.animando_estado = False
        self.progreso.stop()
        self.progreso.set(0)
        self.estado_ia.configure(text="● Lista para conversar")
        self.btn_enviar.configure(state="normal", text="ENVIAR ORDEN")
        self.entry_prompt.focus_set()

    def _animar_estado(self, paso):
        if not self.animando_estado:
            return
        etapas = ("Entendiendo tu solicitud", "Analizando opciones", "Preparando respuesta")
        texto = etapas[(paso // 4) % len(etapas)]
        puntos = "." * (paso % 4)
        self.estado_ia.configure(text=f"● {texto}{puntos}")
        self.after(350, self._animar_estado, paso + 1)

    def _animar_visualizacion(self, nombre, resultado, datos_extra=None):
        canvas = self.visualizaciones.get(nombre)
        if canvas is None or not canvas.winfo_exists():
            return

        estado = self.animaciones.get(nombre, {"id": 0})
        animacion_id = estado["id"] + 1
        datos = {"resultado": resultado, **(datos_extra or {})}
        if nombre == "robotica":
            datos["mapa"] = [
                linea.split() for linea in resultado.splitlines()
                if len(linea.split()) == 8
                and all(celda in {"R", "F", "#", "*", "."} for celda in linea.split())
            ]
        elif nombre == "vision":
            if "matriz" not in datos:
                datos["matriz"] = [
                    [int(valor) for valor in re.findall(r"\d+", linea)]
                    for linea in resultado.splitlines()
                    if len(re.findall(r"\d+", linea)) == 6
                ]
        elif nombre == "pln":
            coincidencia = re.search(r"El mensaje '(.*?)' ha sido clasificado como: (.+)\.", resultado)
            datos["mensaje"] = coincidencia.group(1) if coincidencia else resultado
            datos["clasificacion"] = coincidencia.group(2) if coincidencia else ""
        elif nombre == "experto":
            coincidencia = re.search(r"Análisis de \[(.*?)\]", resultado)
            datos["hechos"] = coincidencia.group(1).replace("'", "").split(", ") if coincidencia else []
            datos["conclusion"] = resultado.split("Conclusión:", 1)[-1].strip()
        elif nombre == "difusa":
            valores = re.search(r"A (-?[\d.]+)°C, la velocidad del ventilador es ([\d.]+)%", resultado)
            datos["temperatura"] = float(valores.group(1)) if valores else 25.0
            datos["velocidad"] = float(valores.group(2)) / 100 if valores else 0.5
            self.control_temperatura.set(max(-10, min(60, datos["temperatura"])))
            self._actualizar_temperatura(datos["temperatura"])
            datos["velocidad"] = ControladorDifuso().calcular(datos["temperatura"]) / 100
        elif nombre == "opt":
            valores = re.search(
                r"x=(-?\d+(?:\.\d+)?), y=(-?\d+(?:\.\d+)?)",
                resultado,
            )
            datos["objetivo"] = (
                (float(valores.group(1)), float(valores.group(2)))
                if valores else (0.0, 0.0)
            )

        self.animaciones[nombre] = {"id": animacion_id, "datos": datos}
        self._paso_visualizacion(nombre, animacion_id, 0)

    def _paso_visualizacion(self, nombre, animacion_id, frame):
        estado = self.animaciones.get(nombre)
        if estado is None or estado["id"] != animacion_id:
            return
        canvas = self.visualizaciones[nombre]
        if not canvas.winfo_exists():
            return
        ancho = max(canvas.winfo_width(), 400)
        alto = max(canvas.winfo_height(), 170)
        datos = estado["datos"]
        canvas.delete("all")

        if nombre == "robotica":
            self._dibujar_mapa(canvas, ancho, alto, datos.get("mapa"), frame)
        elif nombre == "red":
            self._dibujar_red(canvas, ancho, alto, frame)
        elif nombre == "vision":
            self._dibujar_vision(
                canvas,
                ancho,
                alto,
                frame,
                datos.get("matriz"),
                datos.get("original"),
            )
        elif nombre == "difusa":
            self._dibujar_ventilador(
                canvas,
                ancho / 2,
                alto / 2,
                frame,
                datos.get("velocidad", 0.5),
                datos.get("temperatura"),
            )
        elif nombre == "pln":
            self._dibujar_pln(
                canvas, ancho, alto, frame,
                datos.get("mensaje", ""), datos.get("clasificacion")
            )
        elif nombre == "experto":
            self._dibujar_experto(
                canvas, ancho, alto, frame,
                datos.get("hechos", []), datos.get("conclusion", "")
            )
        elif nombre == "opt":
            self._dibujar_optimizacion(
                canvas, ancho, alto, frame, datos.get("objetivo")
            )

        max_frames = 32 if nombre == "vision" and len(datos.get("matriz", [])) > 6 else 80
        if frame < max_frames:
            self.after(55, self._paso_visualizacion, nombre, animacion_id, frame + 1)

    @staticmethod
    def _dibujar_mapa(canvas, ancho, alto, mapa, frame):
        filas = mapa or [
            list("R . . . . . . .".split()),
            list(". # # # . . . .".split()),
            list(". . . . . . . .".split()),
            list(". . . . # . . .".split()),
            list(". # # # . . . .".split()),
            list(". . . . . . . .".split()),
            list(". . . . . . . .".split()),
            list(". . . . . . . F".split()),
        ]
        celda = min((alto - 30) / 8, (ancho - 80) / 8)
        inicio_x, inicio_y = (ancho - celda * 8) / 2, (alto - celda * 8) / 2
        camino = [
            (columna, fila)
            for fila, datos_fila in enumerate(filas)
            for columna, valor in enumerate(datos_fila)
            if valor in {"R", "*", "F"}
        ]
        posicion_robot = camino[min(frame // 4, len(camino) - 1)] if camino else (0, 0)
        for fila, datos_fila in enumerate(filas):
            for columna, valor in enumerate(datos_fila):
                x1, y1 = inicio_x + columna * celda, inicio_y + fila * celda
                x2, y2 = x1 + celda - 3, y1 + celda - 3
                color = "#26333a"
                if valor == "#":
                    color = "#4b5660"
                elif valor == "*":
                    color = "#145a4a"
                elif valor == "F":
                    color = "#146e8a"
                if (columna, fila) == posicion_robot:
                    color = "#40e0ae"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#33434b")
                if valor in {"#", "F"}:
                    canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="■" if valor == "#" else "F",
                        fill="#e8f4f4",
                        font=("Consolas", 10, "bold"),
                    )
                if (columna, fila) == posicion_robot:
                    canvas.create_oval(
                        x1 + 4, y1 + 4, x2 - 4, y2 - 4,
                        fill="#b7ffe9", outline="#ffffff", width=2,
                    )
        canvas.create_text(
            inicio_x, alto - 9, anchor="w",
            text="ROBOT  •  RUTA  •  META  •  OBSTÁCULO",
            fill="#a8c2c8", font=("Consolas", 9),
        )

    @staticmethod
    def _dibujar_red(canvas, ancho, alto, frame):
        capas = [2, 4, 3, 1]
        x_por_capa = [
            ancho * (0.16 + indice * 0.23)
            for indice in range(len(capas))
        ]
        posiciones = []
        for indice, cantidad in enumerate(capas):
            ys = [
                alto / 2 + (n - (cantidad - 1) / 2) * min(34, alto / 7)
                for n in range(cantidad)
            ]
            posiciones.append([(x_por_capa[indice], y) for y in ys])
        for x in range(12, int(ancho), 26):
            for y in range(14, int(alto - 28), 24):
                canvas.create_oval(
                    x, y, x + 1, y + 1,
                    fill="#25434b",
                    outline="",
                )
        for indice in range(len(capas) - 1):
            for nodo_entrada, (x1, y1) in enumerate(posiciones[indice]):
                for nodo_salida, (x2, y2) in enumerate(posiciones[indice + 1]):
                    fase = (frame * 0.12 + nodo_entrada * 0.7 + nodo_salida * 0.35) % 1
                    activo = fase < 0.18
                    canvas.create_line(
                        x1 + 13, y1, x2 - 13, y2,
                        fill="#43f5d0" if activo else "#28515a",
                        width=2 if activo else 1,
                    )
                    pulso_x = x1 + 13 + (x2 - x1 - 26) * fase
                    pulso_y = y1 + (y2 - y1) * fase
                    canvas.create_oval(
                        pulso_x - 3, pulso_y - 3, pulso_x + 3, pulso_y + 3,
                        fill="#b8fff2",
                        outline="#55e6cf",
                    )
        for indice, capa in enumerate(posiciones):
            for nodo, (x, y) in enumerate(capa):
                fase = (frame * 0.1 + nodo * 0.6 + indice * 0.8) % 1
                activo = fase < 0.22
                radio = 11 + (2 if activo else 0)
                canvas.create_oval(
                    x - radio - 5, y - radio - 5, x + radio + 5, y + radio + 5,
                    outline="#238f83" if activo else "#24434b",
                    width=1,
                )
                canvas.create_oval(
                    x - radio, y - radio, x + radio, y + radio,
                    fill="#61f7d0" if activo else "#19333d",
                    outline="#c2fff3" if activo else "#48a5a1",
                    width=2,
                )
                canvas.create_text(
                    x, y + 23, text=f"N{indice}{nodo}",
                    fill="#729ba2", font=("Consolas", 7),
                )
        canvas.create_text(
            18, 17, anchor="w", text=f"NEURAL LINK  //  DATA FLOW {'.' * (frame % 4)}",
            fill="#61e7cd", font=("Consolas", 9, "bold"),
        )
        canvas.create_text(
            ancho / 2, alto - 13, text="ENTRADA        CAPA OCULTA        SALIDA",
            fill="#a8c2c8", font=("Consolas", 9),
        )

    @staticmethod
    def _dibujar_vision(canvas, ancho, alto, frame, matriz, original=None):
        matriz = matriz or [[0, 0, 0, 0, 0, 0] for _ in range(6)]
        filas = len(matriz)
        columnas = len(matriz[0]) if filas else 0
        if not columnas:
            return
        original = original or [[0 for _ in range(columnas)] for _ in range(filas)]
        tamano_celda = min(20, (alto - 42) / filas, (ancho - 100) / (columnas * 2 + 3))
        inicio_x = (ancho - tamano_celda * (columnas * 2 + 3)) / 2
        inicio_y = (alto - tamano_celda * filas) / 2
        barrido = (frame // 3) % columnas
        for fila in range(filas):
            for columna in range(columnas):
                entrada = original[fila][columna]
                borde = matriz[fila][columna]
                gris = max(18, min(220, int(entrada)))
                color_original = f"#{gris // 3:02x}{gris:02x}{min(255, gris + 18):02x}"
                fuerza = max(0, min(255, int(borde)))
                color_borde = (
                    f"#{min(255, 15 + fuerza // 3):02x}"
                    f"{min(255, 55 + fuerza):02x}"
                    f"{min(255, 70 + fuerza // 2):02x}"
                )
                x1 = inicio_x + columna * tamano_celda
                y1 = inicio_y + fila * tamano_celda
                x2 = inicio_x + (columnas + 3 + columna) * tamano_celda
                canvas.create_rectangle(
                    x1, y1, x1 + tamano_celda - 1, y1 + tamano_celda - 1,
                    fill=color_original, outline="#283a41",
                )
                canvas.create_rectangle(
                    x2, y1, x2 + tamano_celda - 1, y1 + tamano_celda - 1,
                    fill=color_borde, outline="#283a41",
                )
        barrido_x = inicio_x + (columnas + 3 + barrido) * tamano_celda
        canvas.create_line(
            barrido_x, inicio_y - 3, barrido_x, inicio_y + tamano_celda * filas,
            fill="#c0fff3", width=2,
        )
        canvas.create_text(
            inicio_x + tamano_celda * columnas / 2,
            alto - 10,
            text="IMAGEN ORIGINAL",
            fill="#a8c2c8", font=("Consolas", 9),
        )
        canvas.create_text(
            inicio_x + tamano_celda * (columnas + 3 + columnas / 2),
            alto - 10,
            text="BORDES SOBEL",
            fill="#a8c2c8", font=("Consolas", 9),
        )

    @staticmethod
    def _dibujar_ventilador(canvas, centro_x, centro_y, frame, velocidad, temperatura=None):
        radio = 52
        canvas.create_oval(
            centro_x - radio, centro_y - radio,
            centro_x + radio, centro_y + radio,
            outline="#37626a", width=2,
        )
        velocidad = max(0, min(1, velocidad))
        vueltas = round(velocidad * 7)
        giro = frame * (0.01 + velocidad * 0.3)
        for indice in range(3):
            angulo = giro + indice * (2 * math.pi / 3)
            punta_x = centro_x + math.cos(angulo) * radio * 0.8
            punta_y = centro_y + math.sin(angulo) * radio * 0.8
            canvas.create_line(
                centro_x, centro_y, punta_x, punta_y,
                fill="#49e0b6", width=5, capstyle="round",
            )
            canvas.create_oval(
                punta_x - 6, punta_y - 6, punta_x + 6, punta_y + 6,
                fill="#76f4d4", outline="",
            )
        canvas.create_oval(
            centro_x - 9, centro_y - 9, centro_x + 9, centro_y + 9,
            fill="#d1fff4", outline="#48c9a7",
        )
        canvas.create_text(
            centro_x, centro_y + 75,
            text=(
                f"{temperatura:.1f}°C   •   POTENCIA {velocidad * 100:.1f}%"
                if temperatura is not None
                else f"POTENCIA {velocidad * 100:.1f}%   •   NIVEL {vueltas}"
            ),
            fill="#b8d6d8", font=("Consolas", 10, "bold"),
        )

    @staticmethod
    def _dibujar_pln(canvas, ancho, alto, frame, mensaje, clasificacion):
        palabras = re.findall(r"\w+", mensaje, flags=re.UNICODE) or ["ESCRIBE", "UN", "MENSAJE"]
        visibles = max(1, frame // 5)
        max_palabras = min(len(palabras), 12)
        visibles = min(visibles, max_palabras)
        paso_x = min(100, (ancho - 60) / max(max_palabras, 1))
        inicio_x = max(20, (ancho - paso_x * max_palabras) / 2)
        for indice, palabra in enumerate(palabras[:max_palabras]):
            x = inicio_x + indice * paso_x
            activo = indice < visibles
            color = "#1d8069" if activo else "#27343a"
            canvas.create_rectangle(
                x, alto / 2 - 16, x + paso_x - 7, alto / 2 + 16,
                fill=color, outline="#4ccca9" if activo else "#40515a",
                width=2 if indice == visibles - 1 and frame % 5 < 3 else 1,
            )
            canvas.create_text(
                x + (paso_x - 7) / 2, alto / 2,
                text=palabra[:13], fill="#effffb" if activo else "#91a5aa",
                font=("Consolas", 9),
            )
        if frame > max_palabras * 5:
            etiqueta = clasificacion or "ANÁLISIS COMPLETADO"
            color = "#9b6132" if "SPAM" in etiqueta.upper() else "#187b60"
            canvas.create_rectangle(
                ancho / 2 - 100, alto - 48, ancho / 2 + 100, alto - 20,
                fill=color, outline="#74eac5",
            )
            canvas.create_text(
                ancho / 2, alto - 34, text=etiqueta[:28],
                fill="#ffffff", font=("Consolas", 10, "bold"),
            )
        canvas.create_text(
            ancho / 2, 18, text="TOKENIZANDO  →  CLASIFICANDO",
            fill="#a8c2c8", font=("Consolas", 9),
        )

    @staticmethod
    def _dibujar_experto(canvas, ancho, alto, frame, hechos, conclusion):
        centros = [ancho * 0.2, ancho * 0.5, ancho * 0.8]
        cy = alto * 0.52
        activo = (frame // 6) % 3
        for indice in range(2):
            canvas.create_line(
                centros[indice] + 60, cy, centros[indice + 1] - 60, cy,
                fill="#4de0b4" if activo == indice else "#435860", width=3,
                arrow="last",
            )
        etiquetas = [
            ", ".join(hechos[:2]) or "HECHOS",
            "REGLAS",
            conclusion[:20] or "CONCLUSIÓN",
        ]
        for indice, etiqueta in enumerate(etiquetas):
            x = centros[indice]
            encendido = activo == indice
            canvas.create_oval(
                x - 54, cy - 30, x + 54, cy + 30,
                fill="#176c59" if encendido else "#26363d",
                outline="#6dffd3" if encendido else "#4d6970", width=2,
            )
            canvas.create_text(
                x, cy, text=etiqueta[:18], fill="#f1fffb",
                font=("Consolas", 9, "bold"),
            )
        pulso_x = centros[activo] + (frame % 6) * 4
        canvas.create_oval(
            pulso_x - 4, cy - 4, pulso_x + 4, cy + 4,
            fill="#fff29a", outline="",
        )
        canvas.create_text(
            ancho / 2, alto - 15, text="HECHOS  →  INFERENCIA  →  RESULTADO",
            fill="#a8c2c8", font=("Consolas", 9),
        )

    @staticmethod
    def _dibujar_optimizacion(canvas, ancho, alto, frame, objetivo):
        centro_x, centro_y = ancho / 2, alto / 2
        escala = min((ancho - 90) / 10, (alto - 50) / 10)
        for unidad in range(-4, 5, 2):
            x = centro_x + unidad * escala
            y = centro_y - unidad * escala
            canvas.create_line(x, 18, x, alto - 18, fill="#29383e")
            canvas.create_line(30, y, ancho - 30, y, fill="#29383e")
        canvas.create_line(30, centro_y, ancho - 30, centro_y, fill="#577078")
        canvas.create_line(centro_x, 18, centro_x, alto - 18, fill="#577078")
        objetivo = objetivo or (0.0, 0.0)
        progreso = min(frame / 65, 1)
        inicio_x, inicio_y = -4.5, 4.0
        actual_x = inicio_x + (objetivo[0] - inicio_x) * progreso
        actual_y = inicio_y + (objetivo[1] - inicio_y) * progreso
        puntos = []
        for indice in range(22):
            t = min((frame / 65) * 22, indice)
            factor = t / 21
            x = inicio_x + (objetivo[0] - inicio_x) * factor
            y = inicio_y + (objetivo[1] - inicio_y) * factor
            puntos.extend((centro_x + x * escala, centro_y - y * escala))
        if len(puntos) >= 4:
            canvas.create_line(*puntos, fill="#35bc96", width=2, smooth=True)
        px, py = centro_x + actual_x * escala, centro_y - actual_y * escala
        canvas.create_oval(
            px - 8, py - 8, px + 8, py + 8,
            fill="#fff08a", outline="#ffffff", width=2,
        )
        canvas.create_text(
            ancho / 2, alto - 12,
            text=f"BUSCANDO MÍNIMO  •  x={actual_x:.2f}  y={actual_y:.2f}",
            fill="#b8d6d8", font=("Consolas", 9),
        )

    def mostrar_resultado_herramienta(self, nombre, resultado):
        modulos = {
            "ejecutar_robotica": ("4.1 Robótica", self.txt_robotica),
            "ejecutar_red_neuronal": ("4.2 Redes Neuronales", self.txt_rn),
            "ejecutar_vision": ("4.3 Visión", self.txt_vision),
            "ejecutar_logica_difusa": ("4.4 Lógica Difusa", self.txt_difusa),
            "ejecutar_pln": ("4.5 PLN", self.txt_pln),
            "ejecutar_sistema_experto": ("4.6 Sist. Expertos", self.txt_experto),
            "ejecutar_algoritmo_genetico": ("Extra: Optimización", self.txt_opt),
            "ejecutar_simulated_annealing": ("Extra: Optimización", self.txt_opt),
        }
        if nombre not in modulos:
            return
        pestaña, consola = modulos[nombre]
        self.tabview.set(pestaña)
        visualizacion = {
            "ejecutar_robotica": "robotica",
            "ejecutar_red_neuronal": "red",
            "ejecutar_vision": "vision",
            "ejecutar_logica_difusa": "difusa",
            "ejecutar_pln": "pln",
            "ejecutar_sistema_experto": "experto",
            "ejecutar_algoritmo_genetico": "opt",
            "ejecutar_simulated_annealing": "opt",
        }[nombre]
        self._animar_visualizacion(visualizacion, resultado)
        lineas = resultado.splitlines() or [resultado]
        self.log_modulo(consola, f"--- {nombre.replace('ejecutar_', '').replace('_', ' ').upper()} ---")
        self.cola_resultados.append((consola, lineas, 0))
        if not self.animando_resultado:
            self.animando_resultado = True
            self._animar_resultado()

    def _animar_resultado(self):
        if not self.cola_resultados:
            self.animando_resultado = False
            return
        consola, lineas, indice = self.cola_resultados[0]
        self.log_modulo(consola, lineas[indice])
        if indice + 1 >= len(lineas):
            self.cola_resultados.pop(0)
            self.after(100, self._animar_resultado)
        else:
            self.cola_resultados[0] = (consola, lineas, indice + 1)
            self.after(65, self._animar_resultado)

    # =========================================================================
    # ACCIONES (FUNCIONES QUE ACTUALIZAN LA INTERFAZ)
    # =========================================================================
    
    def accion_robotica(self):
        self.tabview.set("4.1 Robótica")
        self.log_modulo(self.txt_robotica, "--- EJECUTANDO ROBÓTICA ---")
        obstaculos = [(1,1), (2,1), (3,1), (4,1), (4,2), (4,3), (1,4), (2,4), (3,4)]
        entorno = EntornoRobotica(8, 8, obstaculos)
        camino = AlgoritmoAStar(entorno).buscar((0, 0), (7, 7))
        for y in range(8):
            fila = ""
            for x in range(8):
                if (x, y) in obstaculos: fila += "█ "
                elif camino and (x, y) in camino: fila += "* "
                else: fila += ". "
            self.log_modulo(self.txt_robotica, fila)
        return "Ruta completada."

    def accion_rn(self):
        self.tabview.set("4.2 Redes Neuronales")
        self.log_modulo(self.txt_rn, "--- ENTRENANDO RED NEURONAL ---")
        mlp = RedNeuronalMLP([2, 3, 1])
        X, Y = [[0,0], [0,1], [1,0], [1,1]], [[0], [1], [1], [0]]
        mlp.entrenar(X, Y, epocas=3000, tasa=0.5)
        for x in X:
            pred = mlp.forward(x)[0]
            self.log_modulo(self.txt_rn, f"Entrada: {x} -> Predicción: {pred:.4f}")

    def accion_vision(self):
        self.tabview.set("4.3 Visión")
        self.log_modulo(self.txt_vision, "--- PROCESANDO VISIÓN ---")
        img = [[0]*6 for _ in range(6)]
        for i in range(2, 4):
            for j in range(2, 4): img[i][j] = 255
        bordes = FiltrosImagen.sobel(img)
        for fila in bordes: self.log_modulo(self.txt_vision, " ".join(f"{v:3d}" for v in fila))

    def accion_difusa(self, temperatura: float):
        self.tabview.set("4.4 Lógica Difusa")
        self.log_modulo(self.txt_difusa, f"--- LÓGICA DIFUSA: {temperatura}°C ---")
        vel = ControladorDifuso().calcular(temperatura)
        self.log_modulo(self.txt_difusa, f"Velocidad: {vel:.2f}%")

    def accion_pln(self, mensaje: str):
        self.tabview.set("4.5 PLN")
        self.log_modulo(self.txt_pln, f"--- CLASIFICANDO: '{mensaje}' ---")
        modelo = ClasificadorPLN()
        modelo.entrenar(["Gana dinero rápido", "Oferta increíble"], ["Hola cómo estás", "Reunión mañana"])
        res = modelo.clasificar(mensaje)
        self.log_modulo(self.txt_pln, f"Resultado: {res}")

    def accion_experto(self, sintomas: str):
        self.tabview.set("4.6 Sist. Expertos")
        hechos = [s.strip().lower() for s in sintomas.split(",")]
        self.log_modulo(self.txt_experto, f"--- SÍNTOMAS: {hechos} ---")
        res, pasos = MotorExperto().inferir(hechos)
        for p in pasos: self.log_modulo(self.txt_experto, f"-> {p}")
        diag = [h for h in res if h in ["gripe", "sarampion", "reposo", "hospital"]]
        self.log_modulo(self.txt_experto, f"Diagnóstico: {diag}")

    def accion_ag(self):
        self.tabview.set("Extra: Optimización")
        self.log_modulo(self.txt_opt, "--- EJECUTANDO ALGORITMO GENÉTICO ---")
        x, y = Optimizador.genetico()
        self.log_modulo(self.txt_opt, f"Mejor solución: x={x:.4f}, y={y:.4f}")

    def accion_sa(self):
        self.tabview.set("Extra: Optimización")
        self.log_modulo(self.txt_opt, "--- EJECUTANDO SIMULATED ANNEALING ---")
        x, y = Optimizador.annealing()
        self.log_modulo(self.txt_opt, f"Mejor solución: x={x:.4f}, y={y:.4f}")

if __name__ == "__main__":
    app = AppControlIA()
    app.mainloop()