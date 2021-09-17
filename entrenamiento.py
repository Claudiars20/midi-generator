#librerias
import os, argparse, time
import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Variables y argumentos

DIM_SALIDA = 129

class args:
    # Se agrega un nuevo argumento para DATA_PATH y su valor predeterminado como data/midi
    DATA_PATH = 'data/Huaynos'
    # Se agrega un nuevo argumento para EXPERIMENT_PATH de tipo string y su valor predeterminado como experiments/default
    EXPERIMENTO_PATH = 'experimentos/default'
    # Se agrega un nuevo argumento para TAMANIO_RNN de tipo entero y su valor predeterminado de 64
    TAMANIO_RNN = 64
    # Se agrega un nuevo argumento para NUM_CAPAS de tipo entero y su valor predeterminado de 1
    NUM_CAPAS = 2
    # Se agrega un nuevo argumento para TAZA_APRENDIZAJE de tipo float y su valor predeterminado es 0.001 
    TAZA_APRENDIZAJE = 0.001
    # Se agrega un nuevo argumento para TAMANIO_VENTANA de tipo entero y su valor predeterminado de 20   
    TAMANIO_VENTANA = 20
    # Se agrega un nuevo argumento para TAMANIO_LOTES de tipo entero y su valor predeterminado de 32
    TAMANIO_LOTES = 32
    # Se agrega un nuevo argumento para NUM_EPOCAS de tipo entero y su valor predeterminado de 100
    NUM_EPOCAS = 100
    # Se agrega un nuevo argumento para DROPOUT de tipo float y su valor predeterminado de 0.2
    # porcentaje de pesos que se desactivan en cada paso del conjunto de entrenamiento. Esta es una regularización popular que puede ayudar con el sobreajuste. Los valores recomendados son 0.2-0.5.
    DROPOUT = 0.2
    # Se agrega un nuevo argumento para elegir un algoritmo de optimización donde su valor predeterminado es adam
    # algoritmo de optimización a utilizar. Consulte https://keras.io/optimizers para obtener una lista completa de optimizadores.
    OPTIMIZER = 'adam'
    # Se agrega un nuevo argumento para GRAD_CLIP de tipo float y su valor predeterminado de 5.0
    # recortar degradados en este valor.
    GRAD_CLIP = 5.0
    # Se agrega un nuevo argumento para determinar el número de CPU de tipo entero y su valor predeterminado es de la cantidad que tiene el procesador actual
    CPU = os.cpu_count()
    # Se agrega un nuevo argumento para MAX_FILES_IN_RAM de tipo entero y su valor predeterminado de 128
    ARCHIVOS_MAXIMOS_RAM = 128

# Crear o cargar un modelo guardado
# Devuelve el modelo y el número de época (> 1 si se carga desde el punto de control)
def get_model(args, experiment_dir=None):
    epoca = 0
    if (not experiment_dir):
        modelo = Sequential() # Se crea un modelo secuencial
        # Recorre el número de capas RNN
        for i in range(args.NUM_CAPAS):
            kwargs = dict() # Se crea un diccionario llamado kwargs
            kwargs['units'] = args.TAMANIO_RNN # Se crea una clave llamada units y su valor es el número de neuronas por capa
            # Si esta es la primera capa
            if (i == 0):
                kwargs['input_shape'] = (args.TAMANIO_VENTANA, DIM_SALIDA) # Se crea una clave input_shape y su valor es un par 
                if (args.NUM_CAPAS == 1): # Si el número de capas es 1
                    kwargs['return_sequences'] = False
                else: # Caso contrario
                    kwargs['return_sequences'] = True
                # Agrega una capa LSTM al modelo
                modelo.add(LSTM(**kwargs))
            else:  
                # Si es una capa intermedia
                if (not i == args.NUM_CAPAS - 1):
                    kwargs['return_sequences'] = True # Se crea una clave llamada return_sequences y su valor es True
                    modelo.add(LSTM(**kwargs)) # Agrega una capa LSTM al modelo
                else: # Si es la última capa
                    kwargs['return_sequences'] = False # Se crea una clave llamada return_sequences y su valor es False
                    modelo.add(LSTM(**kwargs)) # Agrega una capa LSTM al modelo
            # Agrega una capa DROPOUT al modelo
            modelo.add(Dropout(args.DROPOUT))
        # Agrega una capa densa con 129 unidades al modelo 
        modelo.add(Dense(DIM_SALIDA))
        # Agrega una función de activación softmax al modelo
        modelo.add(Activation('softmax'))
    else:
        modelo, epoca = utils.load_model_from_checkpoint(experiment_dir) # utilizizacion del método load_model_from_checkpoint de utils.py

    optimizer = Adam() # método de optimización Adam

    # Configuración del proceso de aprendizaje del modelo a través de la función compile()
    modelo.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return modelo, epoca # devuelve el modelo y la cantidad de épocas


def get_callbacks(experiment_dir, checkpoint_monitor='val_accuracy'):
    
    callbacks = []
    
    # guardar puntos de control del modelo
    # .join trabaja para nombrar a los archivos concatenando los strings
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_accuracy_{val_accuracy:.3f}.hdf5')
    print(filepath)
    # objeto se sirve para guardar un modelo o pesos en un intervalo en este caso se enfoca en la eficiencia del modelo
    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='max'))
    # objeto que Reduce la tasa de aprendizaje cuando una métrica ha dejado de mejorar.
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))
    # kit de visualizacion de tensorflow
    callbacks.append(TensorBoard(log_dir=os.path.join(experiment_dir, 'tensorboard-logs'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))
    print(callbacks)
    return callbacks

def main():
    arg = args()
    # obtenemos rutas a los archivos midi que se encuentren en data_path
    try:
        archivos_midi = [os.path.join(arg.DATA_PATH,path) for path in os.listdir(args.DATA_PATH) if '.mid' in path]
        
        #Validamos que haya por menos un archivo midi
        if len(archivos_midi) < 1:
            print('ERROR: no se necontraron archivos midi en {}.'.format(arg.DATA_PATH))
            exit(1)
        print('CORRECTO: se encontraron {} archivos en {}'.format(len(archivos_midi),arg.DATA_PATH))
    except OSError as e:
        print('ERROR: el directorio {} no existe'.format(args.DATA_PATH))
        exit(1)
    
    # creamos el directorio del experimento y devolvemos su nombre
    experimento_path = utils.crear_directorio_experimento(args.EXPERIMENTO_PATH)

    porcentaje_entrenamiento = 0.8 
    indice_separacion = int(float(len(archivos_midi)) * porcentaje_entrenamiento)

    # utilizar generadores para cargar los datos de entrenamiento y validacion de manera diferida
    # asegurando que el usuario no tiene que cargar todos los archivos midi en la RAM a la vez
    generador_entrenamiento = utils.generar_datos(archivos_midi[0:indice_separacion], 
                                               tamanio_ventana=args.TAMANIO_VENTANA,
                                               tamano_bloque=args.TAMANIO_LOTES,
                                               num_hilos=args.CPU,
                                               max_archivos_ram=args.ARCHIVOS_MAXIMOS_RAM)
    generador_validacion = utils.generar_datos(archivos_midi[indice_separacion:], 
                                             tamanio_ventana=args.TAMANIO_VENTANA,
                                             tamano_bloque=args.TAMANIO_LOTES,
                                             num_hilos=args.CPU,
                                             max_archivos_ram=args.ARCHIVOS_MAXIMOS_RAM)                     
    modelo, epocas = get_model(args)
    utils.save_model(modelo, experimento_path)
    print('Modelo guardado en {}'.format(os.path.join(experimento_path, 'model.json')))
    callbacks = get_callbacks(experimento_path)
    # este es un número algo mágico que es el número promedio de ventanas de longitud 20
    # calculado a partir de ~ 5K archivos MIDI del conjunto de datos MIDI de Lakh.
    numero_magico = 827
    tiempo_inicio = time.time()
    modelo.fit(generador_entrenamiento,
                        steps_per_epoch=len(archivos_midi) * numero_magico / args.TAMANIO_LOTES, 
                        epochs=args.NUM_EPOCAS,
                        validation_data=generador_validacion, 
                        validation_steps=len(archivos_midi) * 0.2 * numero_magico / args.TAMANIO_LOTES,
                        callbacks=callbacks,
                        initial_epoch=epocas)
    print('Terminado en {:.2f} segundos'.format(time.time() - tiempo_inicio))

if __name__ == '__main__':
    main()

