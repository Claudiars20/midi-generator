import os, argparse, time
import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

OUTPUT_SIZE = 129 # 0-127 notas + 1 para los vacios

# Argumentos
args = dict()
# Directorio de datos que contiene archivos .midi para usar en el entrenamiento.
args['data_dir'] = 'data/midi'
# Directorio para almacenar los puntos de control y los registros de tensorboard.
args['experiment_dir'] = 'experiments/default'
# Numero de neuronas por capa oculta de RNN
args['rnn_size'] = 64
# Número de capas en RNN
args['num_layers'] = 3
# Tasa de aprendizaje
args['learning_rate'] = None
# Tamaño de ventana
args['window_size'] = 10
# Tamaño de batch
args['batch_size'] = 32
# Numero de epocas
args['num_epochs'] = 10
# Porcentaje de pesos que se desactivan en cada paso del conjunto de entrenamiento.
# Los valores recomendados son 0.2-0.5. - Ayuda con el sobreajuste
args['dropout'] = 0.2
# Números de hilos 
args['n_jobs'] = 8
# Número máximo de archivos midi para cargar en la memoria en la memoria RAM a la vez.
args['max_files_in_ram'] = 25
 

# Crear o cargar un modelo guardado
# Devuelve el modelo y el número de época (> 1 si se carga desde el punto de control)
def get_model(args, experiment_dir=None):
    epoca = 0
    if (not experiment_dir):
        modelo = Sequential() # Se crea un modelo secuencial
        # Recorre el número de capas RNN
        for i in range(args['num_layers']):
            kwargs = dict() # Se crea un diccionario llamado kwargs
            kwargs['units'] = args['rnn_size'] # Se crea una clave llamada units y su valor es el número de neuronas por capa
            # Si esta es la primera capa
            if (i == 0):
                kwargs['input_shape'] = (args['window_size'], OUTPUT_SIZE) # Se crea una clave input_shape y su valor es un par 
                if (args['num_layers'] == 1): # Si el número de capas es 1
                    kwargs['return_sequences'] = False
                else: # Caso contrario
                    kwargs['return_sequences'] = True
                # Agrega una capa LSTM al modelo
                modelo.add(LSTM(**kwargs))
            else:  
                # Si es una capa intermedia
                if (not i == args['num_layers'] - 1):
                    kwargs['return_sequences'] = True # Se crea una clave llamada return_sequences y su valor es True
                    modelo.add(LSTM(**kwargs)) # Agrega una capa LSTM al modelo
                else: # Si es la última capa
                    kwargs['return_sequences'] = False # Se crea una clave llamada return_sequences y su valor es False
                    modelo.add(LSTM(**kwargs)) # Agrega una capa LSTM al modelo
            # Agrega una capa DROPOUT al modelo
            modelo.add(Dropout(args['dropout']))
        # Agrega una capa densa con 129 unidades al modelo 
        modelo.add(Dense(OUTPUT_SIZE))
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
    # Obtener rutas a archivos midi en --data_dir
    midi_files = [os.path.join(args['data_dir'], path) \
                      for path in os.listdir(args['data_dir']) \
                      if '.mid' in path or '.midi' in path]
    
    # Mostramos cuanto archivos midi fueron encontrados
    utils.log(
        'Se encontraron {} archivos midi en {}'.format(len(midi_files), args['data_dir'])
    )
    
    # Validamos que haya por lo menos 1 archivo midi
    if len(midi_files) < 1:
        utils.log(
            'Error: no se encontraron archivos midi en {}. Saliendo.'.format(args['data_dir']))
        exit(1)

    # Creamos el directorio del experimento 
    experiment_dir = utils.experiments_dir(args['experiment_dir'])

    val_split = 0.8 # use el 20 por ciento para la validación
    val_split_index = int(float(len(midi_files)) * val_split)

    # Se utlizar feneradores para cargar los datos de validación / tren de carga diferida, asegurando que el
    # usuario no tiene que cargar todos los archivos midi en la RAM a la vez
    train_generator = utils.generar_data(midi_files[0:val_split_index], 
                                               window_size=args['window_size'],
                                               batch_size=args['batch_size'],
                                               num_threads=args['n_jobs'],
                                               max_files_in_ram=args['max_files_in_ram'])

    val_generator = utils.generar_data(midi_files[val_split_index:], 
                                             window_size=args['window_size'],
                                             batch_size=args['batch_size'],
                                             num_threads=args['n_jobs'],
                                             max_files_in_ram=args['max_files_in_ram'])

    model, epoch = get_model(args)
    print(model.summary())

    utils.save_model(model, experiment_dir)
    utils.log('Modelo guardado en {}'.format(os.path.join(experiment_dir, 'model.json')))

    # Puntos de control
    print(get_callbacks(experiment_dir))
    callbacks = get_callbacks(experiment_dir)
    
    print('Ajustando modelo ...')

    magic_number = 827
    start_time = time.time()
    # Entrenamiento del modelo
    model.fit(train_generator,
                        steps_per_epoch=len(midi_files) * magic_number / args['batch_size'], 
                        epochs=args['num_epochs'],
                        validation_data=val_generator, 
                        validation_steps=len(midi_files) * 0.2 * magic_number / args['batch_size'],
                        callbacks=callbacks,
                        initial_epoch=epoch)
    utils.log('Terminado en {:.2f} segundos'.format(time.time() - start_time))

if __name__ == '__main__':
    main()