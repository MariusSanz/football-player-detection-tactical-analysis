import cv2
import numpy as np
import json

# --- CONFIGURACIÓ DE FITXERS ---
NOM_IMATGE_TELE = 'image1.png' 
NOM_CAMP_BASE = 'camp_base.png'
FITXER_JSON = 'jugadors_defensius.json'
FITXER_SORTIDA = 'resultat_pissarra_tactica.png'

# --- VARIABLES GLOBALS ---
punts_tele = []
punts_mapa = []
jugadors_2d = []
matriu_homografia = None

# --- FUNCIONS DE CONTROL DE RATOLÍ ---
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

def clic_jugadors(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and matriu_homografia is not None:
        # 1. Marcar a la imatge
        cv2.circle(img_jugadors_copy, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow('3. Marca els jugadors', img_jugadors_copy)
        
        # 2. Transformar a 2D
        pt = np.array([[[x, y]]], dtype=np.float32)
        pt_transformat = cv2.perspectiveTransform(pt, matriu_homografia)
        map_x, map_y = pt_transformat[0][0]
        
        # 3. Guardar array
        jugadors_2d.append({"x": float(map_x), "y": float(map_y)})


# ==========================================
# FASE 1: HOMOGRAFIA (4 PUNTS TELE I MAPA)
# ==========================================
img_tele = cv2.imread(NOM_IMATGE_TELE)
img_mapa = cv2.imread(NOM_CAMP_BASE)

if img_tele is None or img_mapa is None:
    print(f"Error: Revisa els noms dels fitxers {NOM_IMATGE_TELE} i {NOM_CAMP_BASE}")
    exit()

img_tele_copy = img_tele.copy()
cv2.namedWindow('1. Referencies (TELE)', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('1. Referencies (TELE)', clic_tele)
cv2.imshow('1. Referencies (TELE)', img_tele_copy)
print("PAS 1: Selecciona 4 punts a la TELE i prem QUALSEVOL TECLA.")
cv2.waitKey(0)
cv2.destroyAllWindows()

img_mapa_copy = img_mapa.copy()
cv2.namedWindow('2. Referencies (MAPA)', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('2. Referencies (MAPA)', clic_mapa)
cv2.imshow('2. Referencies (MAPA)', img_mapa_copy)
print("PAS 2: Selecciona els mateixos 4 punts al MAPA i prem QUALSEVOL TECLA.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(punts_tele) == 4 and len(punts_mapa) == 4:
    matriu_homografia, _ = cv2.findHomography(
        np.array(punts_tele, dtype=np.float32),
        np.array(punts_mapa, dtype=np.float32)
    )
else:
    print("Error: No s'han seleccionat 4 punts. Sortint del programa...")
    exit()

# ==========================================
# FASE 2: MARCATGE MANUAL DE JUGADORS
# ==========================================
img_jugadors_copy = img_tele.copy()
cv2.namedWindow('3. Marca els jugadors', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('3. Marca els jugadors', clic_jugadors)
cv2.imshow('3. Marca els jugadors', img_jugadors_copy)

print("PAS 3: Fes clic ALS PEUS de tots els jugadors defensius.")
print("Quan acabis, prem QUALSEVOL TECLA per continuar.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(jugadors_2d) > 0:
    with open(FITXER_JSON, 'w') as f:
        json.dump(jugadors_2d, f, indent=4)
    print(f"\nDetectats {len(jugadors_2d)} jugadors. Guardat a {FITXER_JSON}.")
else:
    print("\nNo has marcat cap jugador. Sortint del programa...")
    exit()

# ==========================================
# FASE 3: ANÀLISI DE LA FORMACIÓ I DIBUIX
# ==========================================
mapa_resultat = img_mapa.copy()

# Llistes per guardar els jugadors de cada línia
defenses_pts = []
migcampistes_pts = []
atacants_pts = []

if len(jugadors_2d) > 1:
    xs = [j['x'] for j in jugadors_2d]
    min_x, max_x = min(xs), max(xs)
    
    rang_total = max_x - min_x
    terc_rang = rang_total / 3.0
    
    limit_1 = min_x + terc_rang
    limit_2 = min_x + (2 * terc_rang)
    
    # Classificar els jugadors per terços
    for p in jugadors_2d:
        if p['x'] <= limit_1:
            defenses_pts.append(p)
        elif p['x'] <= limit_2:
            migcampistes_pts.append(p)
        else:
            atacants_pts.append(p)

# Dibuixar punts (tots els jugadors)
for p in jugadors_2d:
    cv2.circle(mapa_resultat, (int(p['x']), int(p['y'])), 10, (255, 0, 0), -1) 
    cv2.circle(mapa_resultat, (int(p['x']), int(p['y'])), 12, (255, 255, 255), 2)

# Funció auxiliar per dibuixar les línies tàctiques
def dibuixar_linia_tactica(img, llista_punts, color):
    # Ordenem per la coordenada Y perquè la línia no es creui
    llista_ordenada = sorted(llista_punts, key=lambda p: p['y'])
    if len(llista_ordenada) > 1:
        for i in range(len(llista_ordenada) - 1):
            pt1 = (int(llista_ordenada[i]['x']), int(llista_ordenada[i]['y']))
            pt2 = (int(llista_ordenada[i+1]['x']), int(llista_ordenada[i+1]['y']))
            cv2.line(img, pt1, pt2, color, 3)

# Dibuixem les línies unint els jugadors (només si n'hi ha > 1)
# Pots canviar els colors si vols. Aquí: Groc (Defensa), Cian (Migcamp), Magenta (Atac)
dibuixar_linia_tactica(mapa_resultat, defenses_pts, (0, 255, 255))
dibuixar_linia_tactica(mapa_resultat, migcampistes_pts, (255, 255, 0))
dibuixar_linia_tactica(mapa_resultat, atacants_pts, (255, 0, 255))

# Escriure el text a la cantonada inferior dreta
if len(jugadors_2d) > 1:
    text_formacio = f"Formacio: {len(defenses_pts)}-{len(migcampistes_pts)}-{len(atacants_pts)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.2
    gruix = 3
    
    # Calcular la mida del text per posicionar-lo bé al marge
    (ample_text, alt_text), _ = cv2.getTextSize(text_formacio, font, escala, gruix)
    alt_img, ample_img, _ = mapa_resultat.shape
    
    pos_x = ample_img - ample_text - 30 # 30 píxels de marge
    pos_y = alt_img - 30
    
    # Dibuixem el text amb una petita ombra negra per assegurar que es llegeix sobre la gespa
    cv2.putText(mapa_resultat, text_formacio, (pos_x+2, pos_y+2), font, escala, (0, 0, 0), gruix+2)
    cv2.putText(mapa_resultat, text_formacio, (pos_x, pos_y), font, escala, (255, 255, 255), gruix)

    print(f"\nEsquema calculat: {len(defenses_pts)}-{len(migcampistes_pts)}-{len(atacants_pts)}")

# ==========================================
# FASE 4: MOSTRAR RESULTAT I GUARDAR
# ==========================================
cv2.imshow('RESULTAT FINAL', mapa_resultat)
print("\nPAS 4: Prem 'S' per guardar la imatge tàctica o 'ENTER' per tancar sense guardar.")

while True:
    tecla = cv2.waitKey(0) & 0xFF
    if tecla == ord('s') or tecla == ord('S'):
        cv2.imwrite(FITXER_SORTIDA, mapa_resultat)
        print(f"Imatge guardada com a '{FITXER_SORTIDA}'!")
        break
    elif tecla == 13: # Codi Enter
        print("S'ha tancat sense guardar la imatge.")
        break

cv2.destroyAllWindows()
