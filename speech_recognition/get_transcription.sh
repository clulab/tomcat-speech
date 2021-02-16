### to get tnanscriptions from dir

for file in $1/*wav; do
	filename=$(basename $file)
	transcription=$(python sphinx_transcription.py $file)
	echo -e "${filename}\t${transcription}" >> $2
done
