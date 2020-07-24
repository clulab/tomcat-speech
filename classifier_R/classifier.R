######################### Classifier #########################################
classifier_values_all = read.csv("classifier_values_consolidated.csv",header=TRUE, stringsAsFactors = FALSE)


#mean F0s for each gender: 1=female, 2=male
f0_m = mean(subset(classifier_values_all$f0, classifier_values_all$gender == 2))
f0_f = mean(subset(classifier_values_all$f0, classifier_values_all$gender == 1))

#group data by known and unknown gender
gender_unknown = subset(classifier_values_all, (classifier_values_all$gender == 0 | is.na(classifier_values_all$gender)> 0))
gender_known = subset(classifier_values_all, (classifier_values_all$gender == 1 | classifier_values_all$gender == 2))

# add new column and copy known genders to new column
gender_unknown <- cbind(gender_unknown, gender_predicted = NA) 
gender_known <- cbind(gender_known, gender_predicted = NA) 
gender_known$gender_predicted = gender_known$gender



# classifier
for (row in 1:nrow(gender_unknown)) {
    f0_val <- gender_unknown[row, "f0"]
    diff1 = f0_val-f0_m
    diff2 = f0_val-f0_f
    if(diff1 <= 0 ) {
        gender_unknown[row, "gender_predicted"] <- 2
    }
    else if(diff2 >= 0){ 
        gender_unknown[row, "gender_predicted"] <- 1
    }
    else if(abs(diff2) > abs(diff1)){gender_unknown[row, "gender_predicted"] <- 2}
    else if(abs(diff1) >= abs(diff2)){gender_unknown[row, "gender_predicted"] <- 1}
}

#consolidate rows
classifier_values_final = rbind(gender_unknown,gender_known)

#write


write.csv(classifier_values_final, "classifier_values_complete.csv",row.names=FALSE)

