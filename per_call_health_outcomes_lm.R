# get actual results values and split into bins

# required library
require(dplyr)

#set working directory
setwd("/Volumes/LIvES/")

# read in file with by-call results
results <- read.csv("CallChecklistWithCallIDs.csv")


results$Daily.Average.Fat <- as.numeric(as.character(results$Daily.Average.Fat))
results$Daily.Average.Fiber <- as.numeric(as.character(results$Daily.Average.Fiber))
results$Daily.Average.Fruit.Servings <- as.numeric(as.character(results$Daily.Average.Fruit.Servings))
results$Daily.Average.Step.Count <- as.numeric(as.character(results$Daily.Average.Step.Count))
results$Daily.Average.Vegetable.Servings <- as.numeric(as.character(results$Daily.Average.Vegetable.Servings))
results$Fat.Goal <- as.numeric(as.character(results$Fat.Goal))
results$Fiber.Goal <- as.numeric(as.character(results$Fiber.Goal))
results$Fruit.Goal <- as.numeric(as.character(results$Fruit.Goal))
results$Step.Goal <- as.numeric(as.character(results$Step.Goal))
results$Vegetable.Goal <- as.numeric(as.character(results$Vegetable.Goal))


# summarize results
# calculate difference between scores for each goal
summary <- results %>%
  group_by(call_id, ID, Call.Number, Coach, iso_dates) %>%
  summarize(fat_diff = Daily.Average.Fat - Fat.Goal,
            fib_diff = Daily.Average.Fiber - Fiber.Goal,
            step_diff = Daily.Average.Step.Count - Step.Goal,
            fruit_diff = Daily.Average.Fruit.Servings - Fruit.Goal,
            veg_diff = Daily.Average.Vegetable.Servings - Vegetable.Goal
            )

summary <- summary %>%
  rename(
    sid = ID, 
    call_number = Call.Number,
    coach = Coach
  )

# big goals: 0 = exceeded goal, 1 = met goal, 2 = did not meet goal
# reverse this for FAT (0 = decrease, 1 = no change, 2 = decrease)
summary$fat <- ifelse(summary$fat_diff > 0, 2, 1)
summary$fat <- ifelse(summary$fat_diff < 0, 0, summary$fat)

summary$fib <- ifelse(summary$fib_diff > 0, 0, 1)
summary$fib <- ifelse(summary$fib_diff < 0, 2, summary$fib)

summary$steps <- ifelse(summary$step_diff > 0, 0, 1)
summary$steps <- ifelse(summary$step_diff < 0, 2, summary$steps)

summary$fruit <- ifelse(summary$fruit_diff > 0, 0, 1)
summary$fruit <- ifelse(summary$fruit_diff < 0, 2, summary$fruit)

summary$veg <- ifelse(summary$veg_diff > 0, 0, 1)
summary$veg <- ifelse(summary$veg_diff < 0, 2, summary$veg)

# overall bin score exceeded goal = 0, met goal = 1, failed to meet goal = 2
summary$overall <- NA
  
for (row in 1:nrow(summary)) {
  fat <- summary[row, "fat"]
  fib <- summary[row, "fib"]
  steps <- summary[row, "steps"]
  fruit <- summary[row, "fruit"]
  veg <- summary[row, "veg"]
  
  total <- 5
  
  total <- ifelse(is.na(fat), total - 1, total)
  total <- ifelse(is.na(fib), total - 1, total)
  total <- ifelse(is.na(steps), total - 1, total)
  total <- ifelse(is.na(fruit), total - 1, total)
  total <- ifelse(is.na(veg), total - 1, total)
  
  
  summary[row, "overall"] <- round(sum(fat, fib, steps, fruit, veg, na.rm=TRUE) / total, digits=0)
}


# get an overall bin score of improve = 0, no change = 1, decrease = 2
#summary$overall <- round((summary$fat + summary$fib + summary$steps +
#                                summary$fruit + summary$veg) / 5, digits=0)

# save this to a csv
write.csv(summary, "per-call_outcome-scores.csv", row.names=FALSE)

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

