
# ARGPARSE: Interfaz de linea de comandos, ejm: ls, mkdir
import argparse, os, pdb
import pretty_midi
import train
import utils


## Definir argumentos para nuestra interfaz de linea de comandos
def parse_args():
    # Creamos un ArgumentParse
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    ## Agregamos argumentos ##
    # - Nombre del parametro '--parametro'
    # - type: tipo de la variable
    # - default: valor por defecto
    # - help: descripcion del parametro

    parser.add_argument('--experiment_dir', type=str,
                        default='experiments/default',
                        help='directorio desde el que cargar el modelo guardado. Si se omite, utilizará el directorio más reciente de experiments/.')

    parser.add_argument('--save_dir', type=str,
    					help='directorio para guardar los archivos generados. Se creará un directorio si aún no existe. Si no se especifica, los archivos se guardarán en generated/ dentro --experiment_dir.')
    
    parser.add_argument('--midi_instrument', default='Acoustic Grand Piano',
                        help='Nombre (o número) del instrumento MIDI que se utilizará para los archivos generados. Consulte https://www.midi.org/specifications/item/gm-level-1-sound-set para obtener una lista completa de los nombres de los instrumentos.')
    
    parser.add_argument('--num_files', type=int, default=10,
                        help='número de archivos midi para muestrear.')

    parser.add_argument('--file_length', type=int, default=1000,
    					help='Tamaño de cada archivo, medido en 16th notas.')

    parser.add_argument('--prime_file', type=str,
                        help='Archivos principales generados a partir de archivos midi. Si no se especifica, se utilizarán ventanas aleatorias del conjunto de datos de validación para la siembra.')

    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='directorio de datos que contiene archivos .mid para usar seeding/priming. Obligatorio si no se especifica --prime_file')

    return parser.parse_args()

## Verificar las carpetas del experimento
def get_experiment_dir(experiment_dir):
	
    # si la carpeta de experiments existe, listamos los archivos
	if experiment_dir == 'experiments/default':
		dirs_ = [os.path.join('experiments', d) for d in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', d))]
		experiment_dir = max(dirs_, key=os.path.getmtime)

    # si no existe el archivo model.json, terminamos la ejecucion
	if not os.path.exists(os.path.join(experiment_dir, 'model.json')):
		utils.log('Error: {} no existe. Estas seguro que {} es un experimento valido? .Saliendo.'.format(os.path.join(args.experiment_dir), 'model.json', experiment_dir), True)
		exit(1)

	return experiment_dir


def main():

    # Interfaz de linea de comandos definida
    args = parse_args()
    args.verbose = True

    # Validacion de archivo principal Y datos principales
    if args.prime_file and not os.path.exists(args.prime_file):
    	utils.log('Error: archivo princiapl {} no existe. Saliendo.'.format(args.prime_file), True)
    	exit(1)
    else:
    	if not os.path.isdir(args.data_dir):
    		utils.log('Error: data dir {} no existe. Saliendo.'.format(args.prime_file), True)
    		exit(1)

    # recuperamos los archivos midi generados (.mid , .midi )
    midi_files = [ args.prime_file ] if args.prime_file else [ os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if '.mid' in f or '.midi' in f ]


    experiment_dir = get_experiment_dir(args.experiment_dir)
    utils.log('Usando {} como --experiment_dir'.format(experiment_dir), args.verbose)


    if not args.save_dir:
        args.save_dir = os.path.join(experiment_dir, 'generado')

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        utils.log('Directorio creado {}'.format(args.save_dir), args.verbose)

    model, epoch = train.get_model(args, experiment_dir=experiment_dir)
    utils.log('Modelo cargado desde {}'.format(os.path.join(experiment_dir, 'model.json')), 
              args.verbose)

    window_size = model.layers[0].get_input_shape_at(0)[1]
    seed_generator = utils.get_data_generator(midi_files, 
                                              window_size=window_size,
                                              batch_size=32,
                                              num_threads=16,
                                              max_files_in_ram=10)

    # validar el nombre del instrumento midi
    try:
    	# intente analizar el nombre del instrumento como un int
    	instrument_num = int(args.midi_instrument)
    	if not (instrument_num >= 0 and instrument_num <=127):
    		utils.log('Error: {} no es un instrumento compatible. Los valores numéricos deben ser 0-127. Saliendo'.format(args.midi_instrument), True)
    		exit(1)

    	args.midi_instrument = pretty_midi.program_to_instrument_name(instrument_num)

    except ValueError as err:
    	# si el nombre del instrumento es un string
    	try:
    		# validar que se pueda convertir a un número de programa
    		_ = pretty_midi.instrument_name_to_program(args.midi_instrument)

    	except ValueError as er:
    		utils.log('Error: {} no es un instrumento MIDI general válido. Saliendo.'.format(args.midi_instrument), True)
    		exit(1)

    # generar 10 pistas usando semillas aleatorias
    utils.log('Cargando archivos semilla...', args.verbose)
    X, y = next(seed_generator)
    generated = utils.generate(model, X, window_size, args.file_length, args.num_files, args.midi_instrument)

    for i, midi in enumerate(generated):
        file = os.path.join(args.save_dir, '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        utils.log('escribió un archivo midi en {}'.format(file), True)

if __name__ == '__main__':
    main()