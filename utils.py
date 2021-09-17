from genericpath import isdir
import os, glob, random
import pretty_midi
import numpy as np
from keras.models import model_from_json
from multiprocessing import Pool as ThreadPool


def leer_MIDI(path):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nERROR: no se pudo leer el archivo MIDI %s" % (e, path)))
    return midi

def crear_directorio_experimento(experiment_path):
    '''Este método crea el directorio para el nuevo experimento realizado, si la direccion del experimento no existe, la creamos y sus subfolders'''
    # si el directorio no es el predeterminado y ya existe
    if experiment_path != 'experimentos/default' and os.path.exists(experiment_path):
        raise Exception('ERROR: direccion invalida, ya existe '.format(experiment_path))
    # si el directorio es el predeterminado, creamos un nuevo folder para el experimento
    if experiment_path == 'experimentos/default':
        experimentos = os.listdir('experimentos')
        experimentos = [carpeta for carpeta in experimentos if os.path.isdir(os.path.join('experimentos',carpeta))]

        ultimo_experimento = 0
        for carpeta in experimentos:
            try:
                ultimo_experimento = max(int(carpeta),ultimo_experimento)
            except ValueError as e:
                #ignoramos las carpetas no numericas de experimentos
                pass
        experiment_path = os.path.join('experimentos',str(ultimo_experimento+1).rjust(2,'0'))
    
    os.mkdir(experiment_path)
    print('CORRECTO: se creo la carpeta para el experimento {}'.format(experiment_path))

    os.mkdir(os.path.join(experiment_path, 'checkpoints'))
    print('CORRECTO: se creo la carpeta para los puntos de control {}'.format(os.path.join(experiment_path, 'checkpoints')))

    os.mkdir(os.path.join(experiment_path, 'tensorboard-logs'))
    print('CORRECTO: se creo la carpeta para los mensajes de registro de tensorboard {}'.format(os.path.join(experiment_path, 'tensorboard-logs')))
    return experiment_path

def obtener_porcentaje_monofonico(instrumento_roll):
    matriz = instrumento_roll.T > 0
    notas = np.sum(matriz, axis=1)
    n = np.count_nonzero(notas)
    single = np.count_nonzero(notas == 1)
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else: # no hay notas de ningun tipo
        return 0.0

def filtro_monofonico(pm_instrumentos, porcentaje_monofonico = 0.9):
    '''Modelamos las melodias que son monofonicas conforme al porcentaje dado'''
    return [i for i in pm_instrumentos if obtener_porcentaje_monofonico(i.get_piano_roll()) >= porcentaje_monofonico]

# cargamos los datos MIDI de forma diferida
def generar_datos(MIDI_paths, tamanio_ventana = 20, tamano_bloque = 32, num_hilos=os.cpu_count(), max_archivos_ram=50):

    if(num_hilos>1):
        # cargamos los archivos midi para cada hilo
        pool = ThreadPool(num_hilos) 

    indice_carga = 0
    while True:
        archivos_a_cargar = MIDI_paths[indice_carga:indice_carga+max_archivos_ram]
        # actualizamos el indice para seguir cargando los archivos
        indice_carga = (indice_carga+max_archivos_ram)%len(MIDI_paths)

        if num_hilos > 1:
            MIDI_analizados = pool.map(leer_MIDI, archivos_a_cargar)
        else:
            MIDI_analizados = map(leer_MIDI, archivos_a_cargar)
        datos = obtener_ventanas_instrumentos_monofonicos(MIDI_analizados, tamanio_ventana)
        indice_bloque = 0
        while indice_bloque + tamano_bloque < len(datos[0]):
            res = (datos[0][indice_bloque: indice_bloque + tamano_bloque], 
                   datos[1][indice_bloque: indice_bloque + tamano_bloque])
            yield res
            indice_bloque = indice_bloque + tamano_bloque
        # liberamos la memoria
        del MIDI_analizados 
        del datos 

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        f.write(model.to_json())

# Cargar modelo desde checkpoint
def load_model_from_checkpoint(model_dir):

    '''Carga el modelo con mejor rendimiento'''
    # Lectura del modelo en formato json
    # model_dir = carpeta
    # os.path.join (concatena direcciones)
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())

    # Iteraciones = 0
    epoch = 0
    # Nuevo checkpoint = 
    # - glob.glob()
    # Devuelve una lista posiblemente vacía de nombres de ruta 
    # que coincidan con el nombre de ruta, que debe ser una 
    # cadena que contenga una especificación de ruta. relative (*)
    # - glob.iglob()
    # Devuelve un iterador que produce los mismos valores que glob () 
    # sin almacenarlos todos simultáneamente.
    # Retorna el ultimo archivo
    newest_checkpoint = max(glob.iglob(model_dir + 
    	                    '/checkpoints/*.hdf5'), 
                            key=os.path.getctime)

    # hdf5 - información iteracion
    if newest_checkpoint: 
       # Iteracion
       epoch = newest_checkpoint[-22:-19]
       # model.load_weights = Carga todos los pesos de capa, 
       # ya sea desde un archivo de peso TensorFlow o HDF5.
       # Al cargar pesos en formato HDF5, devuelve none.
       model.load_weights(newest_checkpoint)
    return model, epoch

# Generar
# seed = semilla
def generate(model, seeds, window_size, length, num_to_gen, instrument_name):
    
    # Generar archivo pretty midi desde un modelo usando una semilla
    def _gen(model, seed, window_size, length):
        
        # Arreglo generados
        generated = []
        # ring buffer
        # buffer = copia de semilla como arreglo
        buf = np.copy(seed).tolist()
        # mientras generados sea < tamaño generar
        while len(generated) < length:
            # [[2,2,2...]]
            # Nuevo eje, np.expand_dims(array, posición en los ejes, donde sera expandido)
            arr = np.expand_dims(np.asarray(buf), 0)
            # model.predict =  predecir nuevo 
            pred = model.predict(arr) #<----------------------------------------------------------------------
            
            # argmax sampling (NOT RECOMMENDED), or...
            # index = np.argmax(pred)
            
            # prob distrobuition sampling
            # p = probabilidades de aparicion asociadas a las predicciones
            # Genera datos aleatorios de un arreglo
            # Devuelve una muestra aleatoria
            index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
            # pred es un arreglo de 0 de la cantidad de columnas de la semilla
            pred = np.zeros(seed.shape[1])

            # Cambiamos el valor de pred[i] x 1
            pred[index] = 1
            # Agregamos al arreglos de generados la prediccion
            generated.append(pred)
            # Quitamos el primer valor de la semilla 
            buf.pop(0)
            # Agregamos el valor de la predicción a la semilla
            buf.append(pred)
        # devolvemos generados
        return generated

    midis = []
    # recorremos cant. de gen.
    for i in range(0, num_to_gen):
        # recuperamos un valor aleatoria de las semillas 
        seed = seeds[random.randint(0, len(seeds) - 1)]
        # generamos dato a partir del modelo, la semilla, el tamaño de la ventana y el tamaño
        # gen = datos generados
        gen = _gen(model, seed, window_size, length)
        # añadimos al arreglo de midis, el dato generado procesado
        midis.append(_network_output_to_midi(gen, instrument_name))
    return midis

# crea un pretty midi file con un solo instrumento usando la codificación one-hot
# salida de keras model.predict. 
def _network_output_to_midi(windows, 
                           instrument_name='Acoustic Grand Piano', 
                           allow_represses=False):

    # Creamos un objeto PrettyMIDI 
    midi = pretty_midi.PrettyMIDI()
    # Crear una instancia de instrumento para el instrumento
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    
    cur_note = None # Una nota nula para empezar
    cur_note_start = None
    clock = 0 # Tiempo

    # Iterar sobre los nombres de las notas, las cuales seran convertidas a un numero de nota despues
    for step in windows:
        # Indice del valor máximo del step
        note_num = np.argmax(step) - 1
        
        # Una nota cambia
        if allow_represses or note_num != cur_note:
            
            # si se ha tocado una nota antes y no fue un vacio
            if cur_note is not None and cur_note >= 0:            
                # agregue la última nota, ahora que tenemos su hora de finalización
                note = pretty_midi.Note(velocity=127, 
                                        pitch=int(cur_note), 
                                        start=cur_note_start, 
                                        end=clock)
                
                instrument.notes.append(note)

            # Actualizamos la nota actual
            cur_note = note_num
            cur_note_start = clock

        # Actualizamos el tempo
        clock = clock + 1.0 / 4

    # Añadimos el conjunto de notas del instrumento al midi
    midi.instruments.append(instrument)
    return midi

def obtener_ventanas_instrumentos_monofonicos(MIDI, tamanio_ventana):
    # X conjunto de notas entrada
    # Y nota salida
    X, Y = [],[]
    # para cada nota dentro del archivo midi
    for m in MIDI:
        # verificamos si no es una nota vacia
        if m is not None:
            pistas_melodicas = filtro_monofonico(m.instruments, 0.9)
            # ahora para cada pista
            for pista in pistas_melodicas:
                # si la cantidad de notas es mayor al tamanio de la ventana
                if(len(pista.notes) > tamanio_ventana):
                    # dividimos el conjunto de notas segun el tamanio de nuestra ventana
                    ventana = codificar_ventana_desplazada(pista, tamanio_ventana)
                    # para cada una de nuestras ventanas encontradas
                    for v in ventana:
                        X.append(v[0])
                        Y.append(v[1])
    return (np.asarray(X), np.asarray(Y))



def codificar_ventana_desplazada(pista, window_size):
    '''one-hot codifica una ventana deslizante de notas de un instrumento midi.
    Este enfoque utiliza el método piano roll, donde cada paso en el deslizamiento de
    ventana representa una unidad de tiempo constante (fs = 4, o 1 seg / 4 = 250ms).
    Esto nos permite codificar silencios.
    se espera que la pista sea monofónico.'''
    roll = np.copy(pista.get_piano_roll(fs=4).T)

    # Cortar silencio inicial
    suma = np.sum(roll, axis=1)
    mask = (suma > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # Transformar la velocidad de las notas a 1s
    roll = (roll > 0).astype(float)

    # Agregar caract. 0 a silencio 1 a notas
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows














