import os, sys
from pocketsphinx import Pocketsphinx, get_model_path, get_data_path, AudioFile

model_path = get_model_path()
data_path = get_data_path()

config = {
    'hmm': os.path.join("../data/sphinx_models", 'en-us'),
    'lm': os.path.join("../data/sphinx_models", 'en-us.lm.bin'),
    'dict': os.path.join(model_path, 'cmudict-en-us.dict')
}

# ps = Pocketsphinx(**config)

ps = AudioFile(**config)


filepath = sys.argv[1]
filenames = os.listdir(filepath)

for file in filenames:
	file_full = os.path.join(filepath, file)
	ps.decode(
	    audio_file=file_full,
	    buffer_size=2048,
	    no_search=False,
	    full_utt=False
	)

	# print(ps.segments())
	# print('Detailed segments:', *ps.segments(detailed=True), sep='\n')

	print("%s\t%s" % (file, ps.hypothesis()))
	# print(ps.confidence())
