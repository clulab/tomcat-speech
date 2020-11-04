# This Makefile is a work in progress - it has not been tested yet!

data/study-1_2020.08: scripts/sync_asist_data
	./scripts/sync_asist_data

data/glove.short.300d.punct.txt:
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth: 
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

test:
	python tomcat_speech/data_prep/asist_data/asist_prep.py --prep_type extract_audio_and_aws_text --data_path data/study-1_2020.08
