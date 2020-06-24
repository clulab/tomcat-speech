import sys
import os
import re
import glob

########
# this code takes a filepath as argument, 
# searches for all filenames that end with "video_transcript.txt"
# and splits them at utterance level
# NOTE: This code does not preserve sentence punctuation
########



# test if filepath given works
if __name__ == "__main__":
	if os.path.isdir(sys.argv[1]) == False:
		sys.exit("ERROR: bad directory path")

# create an output folder:
out = sys.argv[1] + "/out"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(out))

### Variable ####
loc = sys.argv[1]

# list all relevant text files
def txt(location):
	fold = location + '/*video_transcript.txt'
	files = glob.glob(fold)
	return(files)


# split utterances by searching for end punctuation:
def text(ls):
	for item in ls:
		nm = item.split(".txt")
		n = str(nm[0])
		name = n.split("/")
		output = out + "/" + name[-1] + "_split.txt"
		with open(item) as f:
			mylist = list(f)
			sp = re.split('\.\s|\?\s|\!\s', mylist[0])
		with open(output, 'w') as file_handler:
			for m in sp:
				file_handler.write("%s\n" % m)


z = txt(loc)
if len(z) > 0:
	text(z)








	
