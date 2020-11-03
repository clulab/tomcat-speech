# This Makefile is a work in progress - it has not been tested yet!

data/study-1_2020.08: scripts/sync_asist_data
	./scripts/sync_asist_data

build/glove.short.300d.punct.txt:
	python tomcat_speech/

build/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth: tomcat_speech/train_and_test_models/train_meld.py \
																	build/glove.short.300d.punct.txt 
	# For the below invocation to work, train_meld.py needs to be able to
	# accept a command line argument sys.argv[1] that specifies the the input
	# glove file.
	python $^ $@

