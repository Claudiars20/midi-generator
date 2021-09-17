
# ARGPARSE: Interfaz de linea de comandos, ejm: ls, mkdir
import argparse, os, pdb
import pretty_midi
import entrenamiento
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
                        default='experimentos/default',
                        help='directorio desde el que cargar el modelo guardado. Si se omite, utilizará el directorio más reciente de experiments/.')

    parser.add_argument('--save_dir', type=str,
    					help='directorio para guardar los archivos generados. Se creará un directorio si aún no existe. Si no se especifica, los archivos se guardarán en generated/ dentro --experiment_dir.')
    
    parser.add_argument('--midi_instrument', default='Acoustic Grand Piano',
                        help='Nombre (o número) del instrumento MIDI que se utilizará para los archivos generados. Consulte https://www.midi.org/specifications/item/gm-level-1-sound-set para obtener una lista completa de los nombres de los instrumentos.')
    
    parser.add_argument('--num_files', type=int, default=10,
                        help='número de archivos midi para muestrear.')

    parser.add_argument('--file_length', type=int, default=500,
    					help='Tamaño de cada archivo, medido en 16th notas.')

    parser.add_argument('--prime_file', type=str,
                        help='Archivos principales generados a partir de archivos midi. Si no se especifica, se utilizarán ventanas aleatorias del conjunto de datos de validación para la siembra.')

    parser.add_argument('--data_dir', type=str, default='data/huaynos',
                        help='directorio de datos que contiene archivos .mid para usar seeding/priming. Obligatorio si no se especifica --prime_file')

    return parser.parse_args()

## Verificar las carpetas del experimento
def get_experiment_dir(experiment_dir):
	
    # si la carpeta de experiments existe, listamos los archivos
	if experiment_dir == 'experimentos/default':
		dirs_ = [os.path.join('experimentos', d) for d in os.listdir('experimentos') if os.path.isdir(os.path.join('experimentos', d))]
		experiment_dir = max(dirs_, key=os.path.getmtime)

    # si no existe el archivo model.json, terminamos la ejecucion
	if not os.path.exists(os.path.join(experiment_dir, 'model.json')):
		print('Error: {} no existe. Estas seguro que {} es un experimento valido? .Saliendo.'.format(os.path.join(args.experiment_dir), 'model.json', experiment_dir), True)
		exit(1)

	return experiment_dir

def main():

    # Interfaz de linea de comandos definida
    args = parse_args()
    args.verbose = True

    # Validacion de archivo principal Y datos principales
    if args.prime_file and not os.path.exists(args.prime_file):
    	print('Error: archivo principal {} no existe. Saliendo.'.format(args.prime_file), True)
    	exit(1)
    else:
    	if not os.path.isdir(args.data_dir):
    		print('Error: data dir {} no existe. Saliendo.'.format(args.prime_file), True)
    		exit(1)

    # recuperamos los archivos midi generados (.mid , .midi )
    archivos_midi = [ args.prime_file ] if args.prime_file else [ os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if '.mid' in f or '.midi' in f ]


    experiment_dir = get_experiment_dir(args.experiment_dir)
    print('Usando {} como --experiment_dir'.format(experiment_dir))


    if not args.save_dir:
        args.save_dir = os.path.join(experiment_dir, 'generado')

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
        print('Directorio creado {}'.format(args.save_dir))

    model, epoch = entrenamiento.get_model(args, experiment_dir=experiment_dir)
    print('Modelo cargado desde {}'.format(os.path.join(experiment_dir, 'model.json')))

    window_size = model.layers[0].get_input_shape_at(0)[1]
    seed_generator = utils.generar_datos(archivos_midi, 
                                              tamanio_ventana=window_size,
                                              tamano_bloque=32,
                                              num_hilos=16,
                                              max_archivos_ram=10)

    # validar el nombre del instrumento midi
    try:
    	# intente analizar el nombre del instrumento como un int
    	instrument_num = int(args.midi_instrument)
    	if not (instrument_num >= 0 and instrument_num <=127):
    		print('Error: {} no es un instrumento compatible. Los valores numéricos deben ser 0-127. Saliendo'.format(args.midi_instrument), True)
    		exit(1)

    	args.midi_instrument = pretty_midi.program_to_instrument_name(instrument_num)

    except ValueError as err:
    	# si el nombre del instrumento es un string
    	try:
    		# validar que se pueda convertir a un número de programa
    		_ = pretty_midi.instrument_name_to_program(args.midi_instrument)

    	except ValueError as er:
    		print('Error: {} no es un instrumento MIDI general válido. Saliendo.'.format(args.midi_instrument), True)
    		exit(1)

    # generar 10 pistas usando semillas aleatorias
    print('Cargando archivos semilla...')
    X, y = next(seed_generator)
    generated = utils.generate(model, X, window_size, args.file_length, args.num_files, args.midi_instrument)

    for i, midi in enumerate(generated):
        file = os.path.join(args.save_dir, '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        print('escribió un archivo midi en {}'.format(file), True)

if __name__ == '__main__':
    main()