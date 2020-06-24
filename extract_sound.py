import sys
import os
import re
import glob

# code must accept a directory (argv[1]), a start time (argv[2]) and an end time (argv[3])
# python path_to_input 00:00:00 00:00:59
############### TEST ##################
# validate input conditions (len(sys.argv) == 4, sys.argv[2] and sys.argv[2]
# are both in HH:MM:SS format, argv[1] is a valid path to directory)
# if any of these conditions fail, exit code with message

# find out if function needed to convert .mp4 to .wav as well?
#######################################

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("ERROR: MISSING ARGUMENTS") 
    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[1]) == False:
    	sys.exit("ERROR: bad directory path")
    elif len(sys.argv) == 4 and os.path.isdir(sys.argv[1]) == True and (re.search(r'[0-9]{2,}\:[0-9]{2}\:[0-9]{2}', sys.argv[2]) == None or re.search(r'[0-9]{2,}\:[0-9]{2}\:[0-9]{2}', sys.argv[3]) == None):
    	sys.exit("invalid time format, use HH:MM:SS")  

### Variables ####
loc = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]


# output folder
# Look for the folder 'output', if it isn't there, create it
out = sys.argv[1] + "/output"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(out))



# function to store all .wav filenames to a list
def wav(location):
	fold = location + '/*.wav'
	files = glob.glob(fold)
	return(files)

ls = []
def extract(list, start_time, end_time ):
	for item in list:
		name = item.split(".wav")
		command = "ffmpeg -i " + item + " -ss " + start_time + " -t " + end_time + " " + name[0] + "_extract.wav"
		command2 = "mv " + name[0] + "_extract.wav" + " " + out
		if (item.endswith(".wav")):
			clip = os.system(command)
			move = os.system(command2)
			ls.append(clip)



y = wav(loc)
if len(y) > 0:
	extract(y,start,end)


######################################

#FFMPEG syntax for .wav files: ffmpeg -i input.wav -ss HH:MM:SS -t HH:MM:SS output.wav 
