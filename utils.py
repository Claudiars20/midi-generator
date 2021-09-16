import os, glob, random
import pretty_midi
import numpy as np
from keras.models import model_from_json
from multiprocessing import Pool as ThreadPool

# Imprimir mensaje
def log(message):
	print('[*] {}'.format(message))

# Container para datos MIDI
def parse_midi(path):
    midi = None
    try:
        # Parsing archivo para manejar MIDI
        midi = pretty_midi.PrettyMIDI(path)
        # Quitar notas invalidas 
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nError leyendo audio MIDI.%s" % (e, path)))
    return midi

# Porcentaje Monofónico
def porcentaje_monofonico(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notas = np.sum(mask, axis=1)
    n = np.count_nonzero(notas)
    haynota = np.count_nonzero(notas == 1)
    if haynota > 0:
        return float(haynota) / float(n)
    elif haynota == 0 and n > 0:
        return 0.0
    else: # No hay notas de ningun tipo
        return 0.0

# Filtar audios monofonicos
def filtrar_monofonicos(pm_instruments, umbral=0.99):
    # Retornar aquellos audios monofonicos
    return [i for i in pm_instruments if porcentaje_monofonico(i.get_piano_roll()) >= umbral]

# Crear directorio de experimentos
def experiments_dir(experiment_dir):
    # Si el directorio del experimento no fue especificado se crea uno númerico
    if experiment_dir == 'experiments/default':
    	experiments = os.listdir('experiments')
    	experiments = [dir_ for dir_ in experiments if os.path.isdir(os.path.join('experiments', dir_))]
    	exp_reciente = 0
    	for direct in experiments:
    		exp_reciente = max(int(direct), exp_reciente)
        # Crear path
    	experiment_dir = os.path.join('experiments', str(exp_reciente + 1).rjust(2, '0'))
    # Experimentos
    os.mkdir(experiment_dir)
    log('Se creó directorio para experimentos {}'.format(experiment_dir))
    # Checkpoints
    os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
    log('Se creó directorio para checkpoints{}'.format(os.path.join(experiment_dir, 'checkpoints')))
    os.mkdir(os.path.join(experiment_dir, 'tensorboard-logs'))
    return experiment_dir

# Cargar data en nucleos
def generar_data(midi_paths, 
                       window_size=20, 
                       batch_size=32,
                       num_threads=8,
                       max_files_in_ram=170):

    if num_threads > 1:
    	# Cargar archivos midi en los procesadores (hilos)
    	pool = ThreadPool(num_threads)

    count = 0

    # Separación de archivos en memoria
    while True:
        load_files = midi_paths[count:count + max_files_in_ram]

        i = (i + max_files_in_ram) % len(midi_paths)

        if num_threads > 1:
       		parsed = pool.map(parse_midi, load_files)
       	else:
       		parsed = map(parse_midi, load_files)

        data = _windows_from_monophonic_instruments(parsed, window_size)
        batch_index = 0
        while batch_index + batch_size < len(data[0]):      
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size

def save_model(model, model_dir):
    # Guarda el modelo en un archivo tipo .json
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
# seed = semilla -> X
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

# devolvemos X, y las ventanas con datos desde un instrumento monofonico

# pistas en un archivo midi 
# windows => ventanas
def _windows_from_monophonic_instruments(midi, window_size):
    # X y -> arreglos
    X, y = [], []
    # para cada nota en midi
    for m in midi:
        # Si no es un vacio
        if m is not None:
            melody_instruments = filtrar_monofonicos(m.instruments, 0.9)
            # Para cada instrumento
            for instrument in melody_instruments:
                # si la cantidad de notas es mayor al tamaño de la ventana
                if len(instrument.notes) > window_size:
                    # 
                    windows = _encode_sliding_windows(instrument, window_size)
                    # para cada ventana en las ventanas
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return (np.asarray(X), np.asarray(y))

# one-hot codifica una ventana deslizante de notas de un instrumento midi.
# Este enfoque utiliza el método piano roll, donde cada paso en el deslizamiento
# ventana representa una unidad de tiempo constante (fs = 4, o 1 seg / 4 = 250ms).
# Esto nos permite codificar silencios.
# se espera que pm_instrument sea monofónico.
def _encode_sliding_windows(pm_instrument, window_size):
    
    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # Cortar silencio inicial
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # Transformar la velocidad de las notas a 1s
    roll = (roll > 0).astype(float)
    
    # Porcentaje de eventos que son silencio
    # s = np.sum(roll, axis=1)
    # num_silence = len(np.where(s == 0)[0])
    # print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

    # Agregar caract. 1 a silencio 0 a notas
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows