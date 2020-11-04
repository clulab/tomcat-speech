# This Makefile is a work in progress - it has not been tested yet!

data/study-1_2020.08: scripts/sync_asist_data
	./scripts/sync_asist_data

data/glove.short.300d.punct.txt:
	mkdir -p data
	curl http://vanga.sista.arizona.edu/tomcat/data/$(@F) -o data/$(@F)

build/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth: tomcat_speech/train_and_test_models/train_meld.py \
																	data/glove.short.300d.punct.txt
	mkdir -p $(@D)
	python $^ $@

