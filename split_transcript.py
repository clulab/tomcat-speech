import sys
import os
import re
import glob

if __name__ == "__main__":
	if os.path.isdir(sys.argv[1]) == False:
		sys.exit("ERROR: bad directory path")

out = sys.argv[1] + "/output"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(out))
### Variables ####
loc = sys.argv[1]

# list all relevant text files
def txt(location):
	fold = location + '/*video_transcript.txt'
	files = glob.glob(fold)
	return(files)

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

		# 
		# 	for i in transcript:
		# 		file_handler.write("{}\n".format(i))

z = txt(loc)
if len(z) > 0:
	text(z)








	
