setwd("/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCoach/firstTestsR/APE_runs_hr")

library(XML)

file.list <- list.files(pattern=".tcx$")
for (file.name in file.list) {
	print(paste("Parsing", file.name))
	file.xml <- xmlToList(file.name)
	laps.xml <- file.xml[[1]][[1]]
	laps <- vector("list", length(laps.xml)-3)
	for (i in 1:length(laps)) {
		time <- as.POSIXct(sapply(laps.xml[[i+1]][[7]], function(x){x$Time}), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
		elevation <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){ifelse(is.null(x$AltitudeMeters), NA, x$AltitudeMeters)}))
		distance <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){ifelse(is.null(x$DistanceMeters), NA, x$DistanceMeters)}))
		speed <- as.numeric(sapply(laps.xml[[i+1]][[7]], function(x){ifelse(is.null(x$Extensions$TPX$Speed), NA, x$Extensions$TPX$Speed)}))
		laps[[i]] <- data.frame(time, elevation, distance, speed)
	}
	dataset.all <- do.call(rbind, laps)
	write.table(dataset.all, sub(".tcx", ".tab", file.name), sep="\t", row.names=FALSE, col.names=TRUE)
}

