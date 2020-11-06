# This Makefile is a work in progress - it has not been tested yet!

all: test

DATA_DIR=data/study-1_2020.08
external/opensmile-3.0:
	./scripts/download_opensmile

$(DATA_DIR): scripts/sync_asist_data
	./scripts/sync_asist_data

.PHONY: $(DATA_DIR)

data/glove.short.300d.punct.txt:
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth:
	mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

M4A_FILES = $(wildcard $(DATA_DIR)/*.m4a)
WAV_FILES = $(patsubst $(DATA_DIR)/%.m4a, build/wav_files/%.wav, $(M4A_FILES))

test: $(DATA_DIR)\
	external/opensmile-3.0\
	data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth\
	data/glove.short.300d.punct.txt
	python tomcat_speech/train_and_test_models/test_asist.py\
		--prep_type extract_audio_and_zoom_text\
		--data_path data/study-1_2020.08\
		--opensmile_path external/opensmile-3.0\
		--media_type m4a\
		--glove_file data/glove.short.300d.punct.txt

# Convert .m4a files to .wav files
build/wav_files/%.wav: $(DATA_DIR)/%.m4a
	mkdir -p $(@D)
	ffmpeg -i $< -ac 1 $@

# Convert .vtt files to .tsv files
build/tsv_files/%.tsv: $(DATA_DIR)/%.vtt
	mkdir -p $(@D)
	scripts/vtt_to_tsv $< $@


VTT_FILES = $(wildcard $(DATA_DIR)/*.vtt)
TSV_FILES = $(patsubst $(DATA_DIR)/%.vtt, build/tsv_files/%.tsv, $(VTT_FILES))

test2: $(TSV_FILES)
	@echo "test"

test3: $(WAV_FILES)
