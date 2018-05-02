setwd("C:\\Users\\Larissa\\Documents\\GitHub\\md")

resolved <- read.table("itemsResolved.csv", header = T, sep = ";")
old <- read.table("itemsOld.csv", header = T, sep = "|")

pid <- resolved$pid
size.new <- resolved$size
size.old <- old$size

size.comparison <- data.frame(c(pid), c(size.old), c(size.new))

colnames(size.comparison) <- c("pid", "old_size", "new_size")

write.table(size.comparison, file = "sizeComparision", row.names = F, col.names = T, sep = "|")
