#--------------------------------------------------------------------------------------------------
setwd("/home/hector/Desktop/work/APE_run/data_corrected")

file.list <- list.files(pattern=".tab$")

datasets.all <- list()
i <- 1
for (file.name in file.list) {
    print(paste("Processing", file.name))
    dataset <- read.table(file.name, header=TRUE, sep="\t")
    dataset.lenght <- nrow(dataset)
    window.size.half <- 3
    slope <- sapply(1:dataset.lenght, 
                    function(x) {
                        index <- (x-window.size.half):(x+window.size.half)
                        index <- index[index > 0]
                        dataset.part <- dataset[index,]
                        model <- lm(elevation ~ distance, dataset.part)
                        coef(model)[2]
                    })
    datasets.all[[i]] <- cbind(dataset[!is.na(dataset$distance),], slope=slope[!is.na(dataset$distance)])
    i <- i + 1
}

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

file.list <- list.files(pattern=".tcx$")
dates <- NULL
for (file.name in file.list) {
	print(paste("Parsing", file.name))
	file.xml <- xmlToList(file.name)
	dates <- c(dates, as.POSIXct(file.xml[[1]][[1]][[1]]))
}

dates.order <- order(dates)

#--------------------------------------------------------------------------------------------------
profile.2D <- function(data.in, res=25, p=c(0.25, 0.5, 0.75)) {
    v.x.factor <- cut(data.in[,1], res)
    v.x.intervals <- sapply(strsplit(levels(v.x.factor), ","), function(x){as.numeric(c(sub("[(]", "", x[1]), sub("[]]", "", x[2])))})
    v.x.mids <- apply(v.x.intervals, 2, function(x){mean(x)})
    v.x.counts <- as.numeric(table(v.x.factor))
    data.split <- split(data.in[,2], v.x.factor)
    q <- sapply(data.split, quantile, probs=p)

    list(stats=q, mids=v.x.mids, intervals=v.x.intervals, counts=v.x.counts)
}
#--------------------------------------------------------------------------------------------------

slope.min <- -0.2                             #slope
slope.max <- 0.2                              #slope
speed.min <- 0                                #speed
speed.max <- 5                                #speed
slope.resolution <- 25
certainty.threshold <- 100
n.files <- length(datasets.all)
runs.min <- 20

from <- seq(1, length(dates)-runs.min+1, 1)
to <- from + runs.min - 1

custom.palette <- colorRamp(c("blue", "red"))
pdf("../modelThroughTime", width=8, height=8, paper="special")
par(oma=c(0,0,0,0), mar=c(4,4,2,1))
plot(0, xlim=c(slope.min, slope.max), ylim=c(2.6, 3.5), main="Evolution of the median speed", xlab="Slope [%]", ylab="Speed [m/s]")
for (model.index in 1:length(from)) {
    training.data <- do.call(rbind, datasets.all[dates.order[from[model.index]]:dates.order[to[model.index]]])
    subset.data <- training.data[,c("slope", "speed")]
    subset.data.filtered <- subset.data[((subset.data[,1] > slope.min) & (subset.data[,1] < slope.max) & (subset.data[,2] > speed.min) & (subset.data[,2] < speed.max)), ]
    current.profile <- profile.2D(subset.data.filtered)
    
    certainty <- current.profile$counts / certainty.threshold
    certainty[certainty>1] <- 1

    lines(current.profile$mids, current.profile$stats[2,], lwd=2, col=rgb(custom.palette(model.index/length(from))/255))
    rect(current.profile$intervals[1,], 2.6, current.profile$intervals[2,], 3.5, density=5*(1-certainty), angle=90, border=NA)
}
grid()
legend("bottom", legend=c("Past", "Present"), fill=c("blue", "red"))
dev.off()


#--------------------------------------------------------------------------------------------------


