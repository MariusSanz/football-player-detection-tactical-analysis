import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import Voronoi, KDTree
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURACIÓ DE FITXERS I MODELS
# ==========================================
NOM_IMATGE_TELE = 'image6.png'      
NOM_CAMP_BASE = 'camp_base.png'     
FITXER_JSON = 'jugadors_defensius.json'
FITXER_SORTIDA = 'resultat_total_pissarra.png'
RUTA_MODELO = 'yolov8x-seg.pt'      

# --- VARIABLES GLOBALS ---
punts_tele = []
punts_mapa = []
matriu_homografia = None
equip_defensiu_id = None
direccio_defensa = None
punt_pilota_tele = None

# Dades unificades dels jugadors
jugadors_detectats = []
id_equipo_A = None
id_equipo_B = None
color_equipo_A = (0, 0, 255) # Vermell
color_equipo_B = (255, 0, 0) # Blau

# ==========================================
# FUNCIONS AUXILIARS I DE RATOLÍ
# ==========================================
def obtenir_color_dominant(recorte):
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([35, 40, 40])
    verde_alto = np.array([85, 255, 255])
    mascara_verde = cv2.inRange(hsv, verde_bajo, verde_alto)
    mascara_jugador = cv2.bitwise_not(mascara_verde)
    pixeles = recorte[mascara_jugador == 255]
    if len(pixeles) < 5: return [0, 0, 0]
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixeles)
    return kmeans.cluster_centers_[0]

def clic_tele(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(punts_tele) < 4:
        punts_tele.append([x, y])
        cv2.circle(img_tele_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_tele_copy, f"P{len(punts_tele)}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('1. Referencies (TELE)', img_tele_copy)

def clic_mapa(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(punts_mapa) < 4:
        punts_mapa.append([x, y])
        cv2.circle(img_mapa_copy, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img_mapa_copy, f"P{len(punts_mapa)}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('2. Referencies (MAPA)', img_mapa_copy)

def dibuixar_jugadors(img_base):
    img_dibuix = img_base.copy()
    for jug in jugadors_detectats:
        x1, y1, x2, y2 = jug['box']
        grupo_id = jug['equip']
        
        if grupo_id == id_equipo_A:
            cv2.rectangle(img_dibuix, (x1, y1), (x2, y2), color_equipo_A, 2)
            cv2.putText(img_dibuix, "Eq 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_equipo_A, 2)
        elif grupo_id == id_equipo_B:
            cv2.rectangle(img_dibuix, (x1, y1), (x2, y2), color_equipo_B, 2)
            cv2.putText(img_dibuix, "Eq 2", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_equipo_B, 2)
    return img_dibuix

def clic_edicio(event, x, y, flags, param):
    global jugadors_detectats
    idx_clicat = -1
    for i, jug in enumerate(jugadors_detectats):
        x1, y1, x2, y2 = jug['box']
        if x1 <= x <= x2 and y1 <= y <= y2:
            idx_clicat = i
            break

    if event == cv2.EVENT_LBUTTONDOWN:
        if idx_clicat != -1:
            eq_actual = jugadors_detectats[idx_clicat]['equip']
            jugadors_detectats[idx_clicat]['equip'] = id_equipo_B if eq_actual == id_equipo_A else id_equipo_A
        else:
            amplada, alcada = 20, 40
            nova_caixa = [x - amplada//2, max(0, y - alcada), x + amplada//2, y]
            jugadors_detectats.append({'box': nova_caixa, 'equip': id_equipo_A})
        img_actualitzada = dibuixar_jugadors(img_tele)
        cv2.putText(img_actualitzada, "MODE EDICIO: Clic Esq: Canviar/Afegir | Clic Dret: Eliminar | ENTER: Sortir", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('MODE EDICIO', img_actualitzada)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if idx_clicat != -1:
            jugadors_detectats.pop(idx_clicat)
            img_actualitzada = dibuixar_jugadors(img_tele)
            cv2.putText(img_actualitzada, "MODE EDICIO: Clic Esq: Canviar/Afegir | Clic Dret: Eliminar | ENTER: Sortir", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('MODE EDICIO', img_actualitzada)

def clic_seleccio_equip(event, x, y, flags, param):
    global equip_defensiu_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for jug in jugadors_detectats:
            x1, y1, x2, y2 = jug['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                equip_defensiu_id = jug['equip']
                print(f"--> Has seleccionat l'Equip {1 if equip_defensiu_id == id_equipo_A else 2} com a DEFENSOR.")
                break

def clic_pilota(event, x, y, flags, param):
    global punt_pilota_tele
    if event == cv2.EVENT_LBUTTONDOWN:
        punt_pilota_tele = (x, y)
        print(f"--> Pilota detectada a la posició: {x}, {y}")
        img_temp = img_seleccio_final.copy()
        cv2.circle(img_temp, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(img_temp, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(img_temp, "Pilota! Prem ENTER per continuar", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('4.5 Selecciona Pilota', img_temp)

# ==========================================
# INICI DEL PROGRAMA
# ==========================================
img_tele = cv2.imread(NOM_IMATGE_TELE)
img_mapa = cv2.imread(NOM_CAMP_BASE)

if img_tele is None or img_mapa is None:
    print(f"❌ Error: Revisa els fitxers {NOM_IMATGE_TELE} i {NOM_CAMP_BASE}")
    exit()

# --- FASE 1: HOMOGRAFIA MANUAL ---
print("\n--- FASE 1: HOMOGRAFIA ---")
img_tele_copy = img_tele.copy()
cv2.namedWindow('1. Referencies (TELE)', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('1. Referencies (TELE)', clic_tele)
cv2.imshow('1. Referencies (TELE)', img_tele_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_mapa_copy = img_mapa.copy()
cv2.namedWindow('2. Referencies (MAPA)', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('2. Referencies (MAPA)', clic_mapa)
cv2.imshow('2. Referencies (MAPA)', img_mapa_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(punts_tele) == 4 and len(punts_mapa) == 4:
    matriu_homografia, _ = cv2.findHomography(
        np.array(punts_tele, dtype=np.float32),
        np.array(punts_mapa, dtype=np.float32)
    )
else:
    exit()

# --- FASE 2: DETECCIÓ YOLO ---
print("\n--- FASE 2: DETECCIÓ YOLO EN CURS ---")
model = YOLO(RUTA_MODELO)
resultados = model.predict(img_tele, conf=0.25, verbose=False)

personas_cajas_tmp = []
personas_colores = []

for box in resultados[0].boxes:
    cls = int(box.cls[0])
    if cls == 0 or cls == 2:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if x1 < 0 or y1 < 0 or x2 > img_tele.shape[1] or y2 > img_tele.shape[0]: continue
        personas_cajas_tmp.append([x1, y1, x2, y2])
        recorte = img_tele[y1:y2, x1:x2]
        personas_colores.append(obtenir_color_dominant(recorte))

kmeans_personas = KMeans(n_clusters=3, n_init=10)
kmeans_personas.fit(personas_colores)
etiquetas_equips = kmeans_personas.labels_

conteo_grupos = {0: 0, 1: 0, 2: 0}
for etiqueta in etiquetas_equips: conteo_grupos[etiqueta] += 1
grupos_ordenados = sorted(conteo_grupos.items(), key=lambda x: x[1], reverse=True)
id_equipo_A = grupos_ordenados[0][0]
id_equipo_B = grupos_ordenados[1][0]

for i, caixa in enumerate(personas_cajas_tmp):
    grup = etiquetas_equips[i]
    if grup in [id_equipo_A, id_equipo_B]:
        jugadors_detectats.append({'box': caixa, 'equip': grup})

# --- FASE 2.5: VALIDACIÓ I EDICIÓ MANUAL ---
img_validacio = dibuixar_jugadors(img_tele)
cv2.putText(img_validacio, "Estan ben detectats? (S per Si / N per No)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Validacio de Deteccions', img_validacio)

necessita_edicio = False
while True:
    tecla = cv2.waitKey(0) & 0xFF
    if tecla in [ord('s'), ord('S'), ord('y'), ord('Y')]:
        break
    elif tecla in [ord('n'), ord('N')]:
        necessita_edicio = True
        break
cv2.destroyAllWindows()

if necessita_edicio:
    cv2.namedWindow('MODE EDICIO', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('MODE EDICIO', clic_edicio)
    img_edicio_inicial = dibuixar_jugadors(img_tele)
    cv2.putText(img_edicio_inicial, "MODE EDICIO: Clic Esq: Canviar/Afegir | Clic Dret: Eliminar | ENTER: Sortir", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow('MODE EDICIO', img_edicio_inicial)
    while True:
        if cv2.waitKey(10) == 13: break
    cv2.destroyAllWindows()

# --- FASE 3: SELECCIÓ D'EQUIP ---
print("\n--- FASE 3: SELECCIÓ DE L'EQUIP DEFENSOR ---")
img_seleccio_final = dibuixar_jugadors(img_tele)
cv2.putText(img_seleccio_final, "Fes CLIC a un jugador de l'equip DEFENSOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

cv2.namedWindow('3. Selecciona Equip', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('3. Selecciona Equip', clic_seleccio_equip)
cv2.imshow('3. Selecciona Equip', img_seleccio_final)
while equip_defensiu_id is None:
    cv2.waitKey(10)

# --- FASE 4: DIRECCIÓ DE DEFENSA ---
print("\n--- FASE 4: DIRECCIÓ DE LA DEFENSA ---")
cv2.putText(img_seleccio_final, "PREM 'E' (Esq) o 'D' (Dreta) al teclat", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('3. Selecciona Equip', img_seleccio_final)

while True:
    tecla = cv2.waitKey(0) & 0xFF
    if tecla in [ord('e'), ord('E')]:
        direccio_defensa = 'E'
        break
    elif tecla in [ord('d'), ord('D')]:
        direccio_defensa = 'D'
        break
cv2.destroyAllWindows()

# --- FASE 4.5: SELECCIÓ DE LA PILOTA ---
print("\n--- FASE 4.5: SELECCIÓ DE LA PILOTA ---")
cv2.putText(img_seleccio_final, "Fes CLIC a la PILOTA (i prem ENTER per confirmar)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.namedWindow('4.5 Selecciona Pilota', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('4.5 Selecciona Pilota', clic_pilota)
cv2.imshow('4.5 Selecciona Pilota', img_seleccio_final)

while True:
    tecla = cv2.waitKey(10) & 0xFF
    if tecla == 13 and punt_pilota_tele is not None:
        break
cv2.destroyAllWindows()

# --- FASE 5: TRANSFORMACIÓ 2D I CÀLCUL DE FORMACIÓ ---
jugadors_2d = []
for jug in jugadors_detectats:
    if jug['equip'] == equip_defensiu_id:
        x1, y1, x2, y2 = jug['box']
        centro_x = (x1 + x2) // 2
        centro_y = y2 
        
        pt = np.array([[[centro_x, centro_y]]], dtype=np.float32)
        pt_transformat = cv2.perspectiveTransform(pt, matriu_homografia)
        map_x, map_y = pt_transformat[0][0]
        jugadors_2d.append([float(map_x), float(map_y)])

pt_pil = np.array([[[punt_pilota_tele[0], punt_pilota_tele[1]]]], dtype=np.float32)
pt_pil_transf = cv2.perspectiveTransform(pt_pil, matriu_homografia)
punt_pilota_2d = (float(pt_pil_transf[0][0][0]), float(pt_pil_transf[0][0][1]))

with open(FITXER_JSON, 'w') as f: json.dump([{"x": p[0], "y": p[1]} for p in jugadors_2d], f, indent=4)

# --- DIBUIXAR LÍNIES TÀCTIQUES ---
mapa_resultat = img_mapa.copy()
grup_1, grup_2, grup_3 = [], [], []

if len(jugadors_2d) > 1:
    xs = [j[0] for j in jugadors_2d]
    min_x, max_x = min(xs), max(xs)
    terc_rang = (max_x - min_x) / 3.0
    limit_1 = min_x + terc_rang
    limit_2 = min_x + (2 * terc_rang)
    
    for p in jugadors_2d:
        if p[0] <= limit_1: grup_1.append({"x": p[0], "y": p[1]})
        elif p[0] <= limit_2: grup_2.append({"x": p[0], "y": p[1]})
        else: grup_3.append({"x": p[0], "y": p[1]})

    if direccio_defensa == 'E':
        defenses_pts, migcampistes_pts, atacants_pts = grup_1, grup_2, grup_3
    else:
        defenses_pts, migcampistes_pts, atacants_pts = grup_3, grup_2, grup_1

def dibuixar_linia_tactica(img, llista_punts, color):
    llista_ordenada = sorted(llista_punts, key=lambda p: p['y'])
    if len(llista_ordenada) > 1:
        for i in range(len(llista_ordenada) - 1):
            pt1 = (int(llista_ordenada[i]['x']), int(llista_ordenada[i]['y']))
            pt2 = (int(llista_ordenada[i+1]['x']), int(llista_ordenada[i+1]['y']))
            cv2.line(img, pt1, pt2, color, 3)

if len(jugadors_2d) > 1:
    dibuixar_linia_tactica(mapa_resultat, defenses_pts, (0, 255, 255))   
    dibuixar_linia_tactica(mapa_resultat, migcampistes_pts, (255, 255, 0)) 
    dibuixar_linia_tactica(mapa_resultat, atacants_pts, (255, 0, 255))   

    text_formacio = f"Formacio: {len(defenses_pts)}-{len(migcampistes_pts)}-{len(atacants_pts)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (ample, alt), _ = cv2.getTextSize(text_formacio, font, 1.2, 3)
    pos_x, pos_y = mapa_resultat.shape[1] - ample - 30, mapa_resultat.shape[0] - 30
    cv2.putText(mapa_resultat, text_formacio, (pos_x+2, pos_y+2), font, 1.2, (0, 0, 0), 5)
    cv2.putText(mapa_resultat, text_formacio, (pos_x, pos_y), font, 1.2, (255, 255, 255), 3)

# ==========================================
# --- FASE 6: VORONOI MULTI-FACTOR (MATPLOTLIB) ---
# ==========================================
print("\n--- FASE 6: GENERANT DIAGRAMA DE VORONOI ---")

if len(jugadors_2d) < 3:
    print("❌ Error: Mínim 3 jugadors per fer Voronoi.")
    exit()

jugadores = np.array(jugadors_2d)
h, w, _ = mapa_resultat.shape

min_x, max_x = np.min(jugadores[:, 0]), np.max(jugadores[:, 0])
min_y, max_y = np.min(jugadores[:, 1]), np.max(jugadores[:, 1])

margen = 100 
rect_x_min = max(0, min_x - margen)
rect_x_max = min(w, max_x + margen)
rect_y_min = max(0, min_y - margen)
rect_y_max = min(h, max_y + margen)

puntos_espejo = []
for x, y in jugadores:
    puntos_espejo.append([2 * rect_x_min - x, y]) 
    puntos_espejo.append([2 * rect_x_max - x, y]) 
    puntos_espejo.append([x, 2 * rect_y_min - y]) 
    puntos_espejo.append([x, 2 * rect_y_max - y]) 
    
puntos_totales = np.vstack([jugadores, puntos_espejo])
vor = Voronoi(puntos_totales)

# CONFIGURAR PORTERIA A DEFENSAR
if direccio_defensa == 'E':
    meta_centro = np.array([0, h/2])
    text_fletxa = (w*0.03, h*0.48)
else:
    meta_centro = np.array([w, h/2])
    text_fletxa = (w*0.85, h*0.48)

# CÀLCUL DE VULNERABILITAT: ESPAI + PORTERIA + PILOTA
step = 5 
grid_x, grid_y = np.mgrid[rect_x_min:rect_x_max:step, rect_y_min:rect_y_max:step]
puntos_grid = np.c_[grid_x.ravel(), grid_y.ravel()]

tree_defensores = KDTree(jugadores)
espacio_libre, _ = tree_defensores.query(puntos_grid)

dist_a_meta = np.linalg.norm(puntos_grid - meta_centro, axis=1)
dist_a_pilota = np.linalg.norm(puntos_grid - np.array(punt_pilota_2d), axis=1)

atenuacion_meta = 500  
atenuacion_pilota = 600  

peso_meta = 1 / (1 + (dist_a_meta / atenuacion_meta)**2)
peso_pilota = 1 / (1 + (dist_a_pilota / atenuacion_pilota)**2)

# LA FÓRMULA MÀGICA: Multipliquem els 3 factors
vulnerabilidad_score = espacio_libre * peso_meta * peso_pilota

idx_max_vulnerabilidad = np.argmax(vulnerabilidad_score)
punto_mas_vulnerable = puntos_grid[idx_max_vulnerabilidad]

distancias_a_jugadores = np.linalg.norm(jugadores - punto_mas_vulnerable, axis=1)
jugador_vulnerable_idx = np.argmin(distancias_a_jugadores)

# VISUALITZACIÓ
fig, ax = plt.subplots(figsize=(12, 7))
mapa_resultat_rgb = cv2.cvtColor(mapa_resultat, cv2.COLOR_BGR2RGB)
ax.imshow(mapa_resultat_rgb)

for i in range(len(jugadores)):
    region_idx = vor.point_region[i]
    region_vertices_indices = vor.regions[region_idx]
    
    if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
        poligono = vor.vertices[region_vertices_indices]
        if i == jugador_vulnerable_idx:
            poly_patch = Polygon(poligono, facecolor='#FFC107', edgecolor='white', alpha=0.6, linewidth=2, zorder=2, label='Zona Crítica (Meta + Pilota)')
        else:
            poly_patch = Polygon(poligono, facecolor='none', edgecolor='white', alpha=0.7, linewidth=1.5, zorder=2)
        ax.add_patch(poly_patch)

# Dibuixar elements
ax.scatter(jugadores[:, 0], jugadores[:, 1], c='red', edgecolors='white', s=100, zorder=5, label='Defensors')
ax.plot(punt_pilota_2d[0], punt_pilota_2d[1], marker='o', color='white', markeredgecolor='black', markersize=12, zorder=6, label='Pilota')

# Fletxa indicador de porteria
ax.annotate("PORTERIA DEFESADA", xy=meta_centro, xytext=text_fletxa, 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7), zorder=6)

ax.set_title("Voronoi Defensiu: Espai + Meta + Pilota", fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, fontsize=11)

ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis('off')
plt.tight_layout()

plt.savefig(FITXER_SORTIDA, bbox_inches='tight', dpi=300)
print(f"\n✅ Procés acabat! Imatge guardada com a '{FITXER_SORTIDA}'")
plt.show()