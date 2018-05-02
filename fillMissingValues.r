library(mice)
setwd("C:\\Users\\Larissa\\Documents\\GitHub\\md")

data_o <- read.table("itemsOld.csv", header = T, sep = "|", na.strings = c("", "NA"))

tempData   <- mice(data_o, meth = "rf", ntree = 10, m = 1, maxit = 1, printFlag = TRUE)
completedData <- complete(tempData,1)
data_o <- completedData

write.table(data_o, "itemsMissingValuesResolved.csv", row.names = F, col.names = T, sep = "|")
