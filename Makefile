# This Makefile is a work in progress - it has not been tested yet!

all: test

DATA_DIR = data/study-1_2020.08
OPENSMILE_DIR = external/opensmile-3.0
OPENSMILE_CONFIG = is09-13/IS10_paraling.conf
GLOVE_FILE = data/glove.short.300d.punct.txt
EMOTION_MODEL = data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth

# Set the type of audio file that needs to be converted (m4a/mp4)
MEDIA_TYPE=m4a

$(OPENSMILE_DIR):
	./scripts/download_opensmile

$(DATA_DIR): scripts/sync_asist_data
	./scripts/sync_asist_data

.PHONY: $(DATA_DIR)

$(GLOVE_FILE):
	@mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@

$(EMOTION_MODEL):
	@mkdir -p $(@D)
	curl http://vanga.sista.arizona.edu/tomcat/$@ -o $@


test: tomcat_speech/train_and_test_models/test_asist.py\
	$(OPENSMILE_DIR)\
	$(EMOTION_MODEL)\
	$(GLOVE_FILE)
	python $<\
		--prep_type extract_audio_and_zoom_text\
		--data_path $(DATA_DIR)\
		--opensmile_path $(OPENSMILE_DIR)\
		--media_type $(MEDIA_TYPE)\
		--glove_file $(GLOVE_FILE)

# Recipe to convert .$(MEDIA_TYPE) files to .wav files
build/wav_files/%.wav: $(DATA_DIR)/%.$(MEDIA_TYPE)
	@mkdir -p $(@D)
	ffmpeg -i $< -ac 1 $@

# Recipe to convert .vtt files to .tsv files
build/tsv_files/%.tsv: scripts/vtt_to_tsv $(DATA_DIR)/%.vtt
	@mkdir -p $(@D)
	$^ $@



# Recipe to create an OpenSMILE output CSV from a .wav file
build/opensmile_output/%.csv: build/wav_files/%.wav
	@mkdir -p $(@D)
	@echo "Extracting features from $< ..."
	@$(OPENSMILE_DIR)/bin/SMILExtract\
		-C $(OPENSMILE_DIR)/config/$(OPENSMILE_CONFIG)\
		-I $<\
		-lldcsvoutput\
		$@ &> /dev/null


# ==================================================
# ASIST-specific portion of the Makefile starts here
# ==================================================

# We only want .vtt files from HSR data
VTT_FILES = $(wildcard $(DATA_DIR)/HSR*.vtt)

AVERAGED_CSV_FILES= $(patsubst build/tsv_files/HSRData_AudioTranscript_%.tsv, build/averaged_csv_files/%.csv, $(TSV_FILES))

TSV_FILES = $(patsubst $(DATA_DIR)/%.vtt, build/tsv_files/%.tsv, $(VTT_FILES))
M4A_FILES = $(wildcard $(DATA_DIR)/*.$(MEDIA_TYPE))
WAV_FILES = $(patsubst $(DATA_DIR)/%.$(MEDIA_TYPE), build/wav_files/%.wav, $(M4A_FILES))

# Defining set of OpenSMILE CSV files
OPENSMILE_CSV_FILES = $(patsubst build/wav_files/%.wav, build/opensmile_output/%.csv, $(WAV_FILES))


build/averaged_csv_files/%.csv: scripts/align_text_and_acoustic_data\
								build/tsv_files/HSRData_AudioTranscript_%.tsv\
								build/opensmile_output/HSRData_Audio_%.csv
	@mkdir -p $(@D)
	$^ $@



tsv_files: $(TSV_FILES)
wav_files: $(WAV_FILES)
opensmile_csv_files: $(OPENSMILE_CSV_FILES)
averaged_csv_files: $(firstword $(AVERAGED_CSV_FILES))
