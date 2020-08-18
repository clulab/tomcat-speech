library(stringr)
############ Read files ####################
setwd("~/github/asist-speech/classifier_R/extra")
speaker = read.csv("speaker2idx.csv",header=TRUE, stringsAsFactors = FALSE) #file with speaker details
names(speaker) <- c("speaker", "Speaker", "gender") #rename columns

## dev ##
mfcc = read.csv("dev_mfcc.csv",header=TRUE, stringsAsFactors = FALSE) #file with mfcc values
filenames = read.csv("filenames_dev.csv",header=TRUE, stringsAsFactors = FALSE) #list of file names in audio directory
utt = read.csv("dev_sent_emo.csv",header=TRUE, stringsAsFactors = FALSE) #list of gold labels

## test ##
mfcc2 = read.csv("test_mfcc.csv",header=TRUE, stringsAsFactors = FALSE)
filenames2 = read.csv("filenames_test.csv",header=TRUE, stringsAsFactors = FALSE) #list of file names in audio directory
utt2 = read.csv("test_sent_emo.csv",header=TRUE, stringsAsFactors = FALSE) #list of gold labels

## train ##
mfcc3 = read.csv("train_mfcc.csv",header=TRUE, stringsAsFactors = FALSE)
filenames3 = read.csv("filenames_train.csv",header=TRUE, stringsAsFactors = FALSE) #list of file names in audio directory
utt3 = read.csv("train_sent_emo.csv",header=TRUE, stringsAsFactors = FALSE) #list of gold labels

############# subsetting #######################

## dev ##
filenames$fileID = str_remove(filenames$fileID, "_IS10.csv")
mfcc = cbind(filenames, mfcc) #add filenames to extracted mfcc values
utt_sub = subset(utt, utt$DiaID_UttID %in% filenames$fileID) #subset gold labels whose mfcc vales are available
audio_na = subset(filenames, !(filenames$fileID %in% utt_sub$DiaID_UttID)) #check for files with mfcc val but no gold labels
mfcc_use = subset(mfcc, mfcc$fileID %in% utt_sub$DiaID_UttID) #mfcc values with gold labels available

## test ##
filenames2$fileID = str_remove(filenames2$fileID, "_IS10.csv")
mfcc2 = cbind(filenames2, mfcc2) #add filenames to extracted mfcc values
utt_sub2 = subset(utt2, utt2$DiaID_UttID %in% filenames2$fileID) #subset gold labels whose mfcc vales are available
audio_na2 = subset(filenames2, !(filenames2$fileID %in% utt_sub2$DiaID_UttID)) #check for files with mfcc val but no gold labels
mfcc_use2 = subset(mfcc2, mfcc2$fileID %in% utt_sub2$DiaID_UttID) #mfcc values with gold labels available

## train ##

filenames3$fileID = str_remove(filenames3$fileID, "_IS10.csv")
mfcc3 = cbind(filenames3, mfcc3) #add filenames to extracted mfcc values
utt_sub3 = subset(utt3, utt3$DiaID_UttID %in% filenames3$fileID) #subset gold labels whose mfcc vales are available
audio_na3 = subset(filenames3, !(filenames3$fileID %in% utt_sub3$DiaID_UttID)) #check for files with mfcc val but no gold labels
mfcc_use3 = subset(mfcc3, mfcc3$fileID %in% utt_sub3$DiaID_UttID) #mfcc values with gold labels available

############ Merging columns for final dataset #################

## dev ##
f0_val = data.frame(cbind(mfcc_use$fileID, mfcc_use$F0final_sma)) 
names(f0_val) <- c(rev(colnames(utt_sub))[1], "f0") #match column names in 2 datasets
mfcc_use = merge(x = utt_sub, y = f0_val, by = "DiaID_UttID", all = TRUE) #merge dataframes such that column values match
classifier_values = merge(x=mfcc_use, y= speaker, by = "Speaker", all.x = TRUE ) #merge data frames such that only rows with available data are merged
names(classifier_values)[names(classifier_values)=="speaker"] <- "speaker_name"
names(classifier_values)[names(classifier_values)=="Speaker"] <- "speaker_ID"
classifier_values$f0 <- as.numeric(as.character(unlist(classifier_values$f0)))

## test ##
f0_val2 = data.frame(cbind(mfcc_use2$fileID, mfcc_use2$F0final_sma)) 
names(f0_val2) <- c(rev(colnames(utt_sub2))[1], "f0") #match column names in 2 datasets
mfcc_use2 = merge(x = utt_sub2, y = f0_val2, by = "DiaID_UttID", all = TRUE) #merge dataframes such that column values match
classifier_values2 = merge(x=mfcc_use2, y= speaker, by = "Speaker", all.x = TRUE ) #merge data frames such that only rows with available data are merged
names(classifier_values2)[names(classifier_values2)=="speaker"] <- "speaker_name"
names(classifier_values2)[names(classifier_values2)=="Speaker"] <- "speaker_ID"
classifier_values2$f0 <- as.numeric(as.character(unlist(classifier_values2$f0)))       

## train ##
f0_val3 = data.frame(cbind(mfcc_use3$fileID, mfcc_use3$F0final_sma)) 
names(f0_val3) <- c(rev(colnames(utt_sub3))[1], "f0") #match column names in f0_val with last column in utt_sub
mfcc_use3 = merge(x = utt_sub3, y = f0_val3, by = "DiaID_UttID", all = TRUE) #merge dataframes such that column values match
classifier_values3 = merge(x=mfcc_use3, y= speaker, by = "Speaker", all.x = TRUE ) #merge data frames such that only rows with available data are merged
names(classifier_values3)[names(classifier_values3)=="speaker"] <- "speaker_name"
names(classifier_values3)[names(classifier_values3)=="Speaker"] <- "speaker_ID"
classifier_values3$f0 <- as.numeric(as.character(unlist(classifier_values3$f0)))  

# write files:
write.csv(classifier_values, "dev_classifier_data.csv",row.names=FALSE)
write.csv(classifier_values2, "test_classifier_data.csv",row.names=FALSE)
write.csv(classifier_values3, "train_classifier_data.csv",row.names=FALSE)

classifier_values_all = rbind(classifier_values, classifier_values2, classifier_values3) #consolidate
write.csv(classifier_values_all, "classifier_values_consolidated.csv",row.names=FALSE)

