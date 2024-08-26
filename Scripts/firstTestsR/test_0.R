#Parsing the .tcx files

setwd("/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCouch/smartCOACH/firstTestsR")

library(XML)
test <- xmlToList("data_corrected/activity_367230665.gpx")
dataset <- sapply(test[[2]][[2]], function(x){as.numeric(x$ele)})

#--------------------------------------------------------------------------------------------------

test <- xmlToList("data_corrected/activity_367230665.tcx")
test[[1]][[1]][[2]][[7]]

time <- as.POSIXct(sapply(test[[1]][[1]][[2]][[7]], function(x){x[[1]]}), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
elevation <- as.numeric(sapply(test[[1]][[1]][[2]][[7]], function(x){x[[3]]}))
distance <- as.numeric(sapply(test[[1]][[1]][[2]][[7]], function(x){x[[4]]}))
speed <- as.numeric(sapply(test[[1]][[1]][[2]][[7]], function(x){x[[5]][[1]][[1]]}))

par(mfrow=c(4,1))
plot(time, type="l", main="Time")
plot(distance, type="l", main="Distance")
plot(elevation, type="l", main="Elevation")
plot(speed, type="l", main="Speed")

#--------------------------------------------------------------------------------------------------

file.xml <- xmlToList("data_corrected/activity_367230665.tcx")
laps.xml <- file.xml[[1]][[1]]
laps <- vector("list", length(laps.xml)-3)
for (i in 1:length(laps)) {
	time <- as.POSIXct(sapply(laps.xml[[i+1]][[7]], function(x){x[[1]]}), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
	elevation <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){x[[3]]}))
	distance <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){x[[4]]}))
	speed <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){x[[5]][[1]][[1]]}))
	laps[[i]] <- data.frame(time, elevation, distance, speed)
}

#par(mfcol=c(4, length(laps)))
#for (l in laps) {
#	plot(l$time, type="l", main="Time")
#	plot(l$distance, type="l", main="Distance")
#	plot(l$elevation, type="l", main="Elevation")
#	plot(l$speed, type="l", main="Speed")
#}


#for (l in laps) {
#	quartz()
#	par(mfcol=c(4, 1))
#	plot(l$time, type="l", main="Time")
#	plot(l$distance, type="l", main="Distance")
#	plot(l$elevation, type="l", main="Elevation")
#	plot(l$speed, type="l", main="Speed")
#}

#--------------------------------------------------------------------------------------------------

dataset.all <- do.call(rbind, laps)

par(mfrow=c(4,1))
plot(dataset.all$time, type="l", main="Time")
grid()
plot(dataset.all$distance, type="l", main="Distance")
grid()
plot(dataset.all$elevation, type="l", main="Elevation")
grid()
plot(dataset.all$speed, type="l", main="Speed")
grid()

#--------------------------------------------------------------------------------------------------
setwd("/home/hector/ownCloud/Shared/SmartDevicesLab/smartCOACH/firstTestsR/data_corrected")
library(XML)

file.list <- list.files(pattern=".tcx$")
dates <- NULL
for (file.name in file.list) {
	print(paste("Parsing", file.name))
	file.xml <- xmlToList(file.name)
	dates <- c(dates, as.POSIXct(file.xml[[1]][[1]][[1]]))
}



