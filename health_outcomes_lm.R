# get actual results values and split into bins

# required library
require(dplyr)

# set working directory
setwd("/Volumes/LIvES/323_files/")

# read in file with all results
results <- read.csv("BL-24M_updated.csv")

# summarize results
# calculate difference between scores for each goal
summary <- results %>%
  group_by(sid) %>%
  summarize(tfat1to2 = (tfat2 - tfat1), tfat1to4 = (tfat4 - tfat1),
            tfib1to2 = (tfib2 - tfib1), tfib1to4 = (tfib2 - tfib1),
            totmet1to2 = (totmet_hrs_weekx_GE3MET_2 - totmet_hrs_weekx_GE3MET_1),
            totmet1to4 = (totmet_hrs_weekx_GE3MET_4 - totmet_hrs_weekx_GE3MET_1),
            fruit1to2 = (fruit_serv_whole_2 - fruit_serv_whole_1),
            fruit1to4 = (fruit_serv_whole_4 - fruit_serv_whole_1),
            veg1to2 = (veg_serv_whole_2 - veg_serv_whole_1),
            veg1to4 = (veg_serv_whole_4 - veg_serv_whole_1))

# create a new dataframe for just 6 month check-in
to2 <- c("sid", "tfat1to2", "tfib1to2", "totmet1to2", "fruit1to2", "veg1to2")
summary1to2 <- summary[to2]
summary1to2 <- na.omit(summary1to2)

# bin each goal; 0 = increase, 1 = no change, 2 = decrease
# reverse this for FAT (0 = decrease, 1 = no change, 2 = decrease)
summary1to2$fat <- ifelse(summary1to2$tfat1to2 > 0, 2, 1)
summary1to2$fat <- ifelse(summary1to2$tfat1to2 < 0, 0, summary1to2$fat)

summary1to2$fib <- ifelse(summary1to2$tfib1to2 > 0, 0, 1)
summary1to2$fib <- ifelse(summary1to2$tfib1to2 < 0, 2, summary1to2$fib)

summary1to2$met <- ifelse(summary1to2$totmet1to2 > 0, 0, 1)
summary1to2$met <- ifelse(summary1to2$totmet1to2 < 0, 2, summary1to2$met)

summary1to2$fruit <- ifelse(summary1to2$fruit1to2 > 0, 0, 1)
summary1to2$fruit <- ifelse(summary1to2$fruit1to2 < 0, 2, summary1to2$fruit)

summary1to2$veg <- ifelse(summary1to2$veg1to2 > 0, 0, 1)
summary1to2$veg <- ifelse(summary1to2$veg1to2 < 0, 2, summary1to2$veg)

# get an overall bin score of improve = 0, no change = 1, decrease = 2
summary1to2$overall <- round((summary1to2$fat + summary1to2$fib + summary1to2$met +
                                summary1to2$fruit + summary1to2$veg) / 5, digits=0)

# create df of just sid and overall score
final <- c("sid", "overall")
point2_score <- summary1to2[final]

# save this to a csv to use for quick calling of outcomes as ys
write.csv(point2_score, "scores_at_point_2.csv", row.names = FALSE)

# create a new dataframe for just final check-in
to4 <- c("sid", "tfat1to4", "tfib1to4", "totmet1to4", "fruit1to4", "veg1to4")
summary1to4 <- summary[to4]
summary1to4 <- na.omit(summary1to4)

# bin each goal; 0 = increase, 1 = no change, 2 = decrease
# reverse this for FAT (0 = decrease, 1 = no change, 2 = decrease)
summary1to4$fat <- ifelse(summary1to4$tfat1to4 > 0, 2, 1)
summary1to4$fat <- ifelse(summary1to4$tfat1to4 < 0, 0, summary1to4$fat)

summary1to4$fib <- ifelse(summary1to4$tfib1to4 > 0, 0, 1)
summary1to4$fib <- ifelse(summary1to4$tfib1to4 < 0, 2, summary1to4$fib)

summary1to4$met <- ifelse(summary1to4$totmet1to4 > 0, 0, 1)
summary1to4$met <- ifelse(summary1to4$totmet1to4 < 0, 2, summary1to4$met)

summary1to4$fruit <- ifelse(summary1to4$fruit1to4 > 0, 0, 1)
summary1to4$fruit <- ifelse(summary1to4$fruit1to4 < 0, 2, summary1to4$fruit)

summary1to4$veg <- ifelse(summary1to4$veg1to4 > 0, 0, 1)
summary1to4$veg <- ifelse(summary1to4$veg1to4 < 0, 2, summary1to4$veg)

# get an overall bin score of improve = 0, no change = 1, decrease = 2
summary1to4$overall <- round((summary1to4$fat + summary1to4$fib + summary1to4$met +
  summary1to4$fruit + summary1to4$veg) / 5, digits=0)

# create df of just sid and overall score
point4_score <- summary1to4[final]

# save this to a csv to use for quick calling of outcomes as ys
write.csv(point4_score, "scores_at_point_4.csv", row.names = FALSE)

############ BY GOAL

# create new df for 6 month totmet
to2tot_met <- c("sid", "totmet1to2")
summary1to2totmet <- summary[to2tot_met]
summary1to2totmet <- na.omit(summary1to2totmet)

summary1to2totmet$met <- ifelse(summary1to2totmet$totmet1to2 > 0, 0, 1)

metfinal <- c("sid", "met")
point2_metscore <- summary1to2totmet[metfinal]
write.csv(point2_metscore, "met_scores_at_point_2.csv", row.names = FALSE)

# fiber
to2fiber <- c("sid", "tfib1to2")
summary1to2fiber <- summary[to2fiber]
summary1to2fiber <- na.omit(summary1to2fiber)

summary1to2fiber$fiber <- ifelse(summary1to2fiber$tfib1to2 > 0, 0, 1)

fibfinal <- c("sid", "fiber")
point2_fibscore <- summary1to2fiber[fibfinal]
write.csv(point2_fibscore, "fiber_scores_at_point_2.csv", row.names = FALSE)

# fruit
to2fruit <- c("sid", "fruit1to2")
summary1to2fruit <- summary[to2fruit]
summary1to2fruit <- na.omit(summary1to2fruit)

summary1to2fruit$fruit <- ifelse(summary1to2fruit$fruit1to2 > 0, 0, 1)

fruitfinal <- c("sid", "fruit")
point2_fruitscore <- summary1to2fruit[fruitfinal]
write.csv(point2_fruitscore, "fruit_scores_at_point_2.csv", row.names = FALSE)

# veg
to2veg <- c("sid", "veg1to2")
summary1to2veg <- summary[to2veg]
summary1to2veg <- na.omit(summary1to2veg)

summary1to2veg$veg <- ifelse(summary1to2veg$veg1to2 > 0, 0, 1)

vegfinal <- c("sid", "veg")
point2_vegscore <- summary1to2veg[vegfinal]
write.csv(point2_vegscore, "veg_scores_at_point_2.csv", row.names = FALSE)

# fat
to2fat <- c("sid", "tfat1to2")
summary1to2fat <- summary[to2fat]
summary1to2fat <- na.omit(summary1to2fat)

summary1to2fat$fat <- ifelse(summary1to2fat$tfat1to2 > 0, 1, 0) #reverse bc increase in fat is bad

fatfinal <- c("sid", "fat")
point2_fatscore <- summary1to2fat[fatfinal]
write.csv(point2_fatscore, "fat_scores_at_point_2.csv", row.names=FALSE)

