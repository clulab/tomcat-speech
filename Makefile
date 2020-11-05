# This Makefile is a work in progress - it has not been tested yet!

external/opensmile-3.0:
	./scripts/download_opensmile

data/study-1_2020.08: scripts/sync_asist_data
	./scripts/sync_asist_data

.PHONY: data/study-1_2020.08

data/glove.short.300d.punct.txt:
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth:
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

test: data/study-1_2020.08\
	external/opensmile-3.0\
	data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth
	python tomcat_speech/data_prep/asist_data/asist_prep.py\
		--prep_type extract_audio_and_zoom_text\
		--data_path data/study-1_2020.08\
		--opensmile_path external/opensmile-3.0
