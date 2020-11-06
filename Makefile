# This Makefile is a work in progress - it has not been tested yet!

all: test

DATA_DIR=data/study-1_2020.08
OPENSMILE_DIR=external/opensmile-3.0
$(OPENSMILE_DIR):
	./scripts/download_opensmile

$(DATA_DIR): scripts/sync_asist_data
	./scripts/sync_asist_data

.PHONY: $(DATA_DIR)

data/glove.short.300d.punct.txt:
	@mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth:
	@mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@


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

# Recipe to convert .m4a files to .wav files
build/wav_files/%.wav: $(DATA_DIR)/%.m4a
	@mkdir -p $(@D)
	ffmpeg -i $< -ac 1 $@

# Recipe to convert .vtt files to .tsv files
build/tsv_files/%.tsv: scripts/vtt_to_tsv $(DATA_DIR)/%.vtt
	@mkdir -p $(@D)
	$^ $@



# Recipe to create an OpenSMILE output CSV from a .wav file
build/opensmile_output/%.csv: build/wav_files/%.wav
	@mkdir -p $(@D)
	$(OPENSMILE_DIR)/bin/SMILExtract\
		-C $(OPENSMILE_DIR)/config/is09-13/IS10_paraling.conf\
		-I $<\
		-lldcsvoutput\
		$@


# ==================================================
# ASIST-specific portion of the Makefile starts here
# ==================================================

# We only want .vtt files from HSR data
VTT_FILES = $(wildcard $(DATA_DIR)/HSR*.vtt)

AVERAGED_CSV_FILES= $(patsubst build/tsv_files/HSRData_AudioTranscript_%.tsv, build/averaged_csv_files/%.csv, $(TSV_FILES))

TSV_FILES = $(patsubst $(DATA_DIR)/%.vtt, build/tsv_files/%.tsv, $(VTT_FILES))
M4A_FILES = $(wildcard $(DATA_DIR)/*.m4a)
WAV_FILES = $(patsubst $(DATA_DIR)/%.m4a, build/wav_files/%.wav, $(M4A_FILES))

# Defining set of OpenSMILE CSV files
OPENSMILE_CSV_FILES = $(patsubst build/wav_files/%.wav, build/opensmile_output/%.csv, $(WAV_FILES))


build/averaged_csv_files/%.csv: scripts/align_text_and_acoustic_data\
								build/tsv_files/HSRData_AudioTranscript_%.tsv\
								build/opensmile_output/HSRData_Audio_%.csv
	mkdir -p $(@D)
	$^ $@



tsv_files: $(TSV_FILES)
wav_files: $(WAV_FILES)
opensmile_csv_files: $(OPENSMILE_CSV_FILES)
averaged_csv_files: $(firstword $(AVERAGED_CSV_FILES))
