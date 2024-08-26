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
distance.step <- 100
n.files <- length(datasets.all)
probabilities <- seq(0.1, 0.9, 0.01)

training.data <- do.call(rbind, datasets.all[1:25])
subset.data <- training.data[,c("slope", "speed")]
subset.data.filtered <- subset.data[((subset.data[,1] > slope.min) & (subset.data[,1] < slope.max) & (subset.data[,2] > speed.min) & (subset.data[,2] < speed.max)), ]
current.profile <- profile.2D(subset.data.filtered, p=probabilities)

run.out <- 26
validation.data <- datasets.all[[run.out]]

slopes <- validation.data$slope
distances <- validation.data$distance
distance.diffs <- c(distances[1], distances[-1] - distances[-length(distances)])

speed.prediction.table <- NULL
for (i in 1:nrow(current.profile$stats)) {
    temp <- sapply(slopes, function(x, q){  temp <- current.profile$stats[q,((x > current.profile$intervals[1,])&(x <= current.profile$intervals[2,]))]
                                            if (length(temp) == 0) {
                                                if (x <= current.profile$intervals[1,1]) {
                                                    temp <- current.profile$stats[q,1]
                                                }
                                                if (x > current.profile$intervals[2,ncol(current.profile$intervals)]) {
                                                    temp <- current.profile$stats[q,ncol(current.profile$stats)]
                                                }
                                            }
                                            temp
                                         }, q=i)
    speed.prediction.table <- cbind(speed.prediction.table, temp)
}

times.prediction.table <- matrix(distance.diffs, nrow=nrow(speed.prediction.table), ncol=ncol(speed.prediction.table)) / speed.prediction.table
times.prediction.start <- colSums(times.prediction.table) / 60

pdf("../estimatedTime_vs_effort.pdf", width=8, height=8, paper="special")
par(oma=c(0,0,0,0), mar=c(4,4,2,1))
plot(probabilities+0.5, times.prediction.start, type="l", main="Estimated time", xlab="Effort", ylab="Time [min]")
grid()
dev.off()

effort.index <- 56
print(paste("You want to finish in", format(times.prediction.start[effort.index], digits=1, nsmall=1), "minutes"))
print(paste("The application will set the effort to", format(probabilities[effort.index]+0.5, digits=1, nsmall=1)))

speed.prediction <- speed.prediction.table[,effort.index]

times.prediction <- distance.diffs / speed.prediction
time.runner <- as.numeric(as.POSIXct(validation.data$time))
time.runner <- time.runner - time.runner[1]

times.prediction.total <- sapply(1:length(times.prediction), function(x){time.runner[x] + sum(times.prediction[-(1:x)])}) / 60
times.prediction.to.goal <- sapply(1:length(times.prediction), function(x){sum(times.prediction[-(1:x)])}) / 60

library("jpeg")
runner <- readJPEG("../runner_woman_small.jpg")

laps <- seq(0, 10000, 2000)
laps.times.total <- sapply(laps, function(x){times.prediction.total[min(which(validation.data$distance>x))]})
laps.times.to.goal <- sapply(laps, function(x){times.prediction.to.goal[min(which(validation.data$distance>x))]})
time.min <- min(times.prediction.total)
time.max <- max(times.prediction.total)
text.coord <- seq(time.min, time.max, length.out=10)

pdf("../estimatedTimeRun26_user.pdf", width=10, height=8, paper="special")
par(mfrow=c(3,1), oma=c(0,0,0,0), mar=c(4,4,3,2))
plot(validation.data$distance, validation.data$elevation, type="l", xlab="Distance [m]", ylab="Elevation [m]", main="Elevation")
abline(v=laps, col="orange")
grid()
plot(validation.data$distance, times.prediction.total, type="l", xlab="Distance [m]", ylab="Time [min]", main=paste("Estimated time (Effort=", format(probabilities[effort.index]+0.5, digits=2, nsmall=2), ")", sep=""))
abline(v=laps, col="orange")
for (x in laps) {rasterImage(runner, x-500, text.coord[2], x, text.coord[5])}
text(laps, text.coord[5], labels="Total time:", pos=4)
text(laps, text.coord[4], labels=paste(format(laps.times.total,digits=1,nsmall=1), "min"), pos=4, col="red", offset=2)
text(laps, text.coord[3], labels="Time to goal:", pos=4)
text(laps, text.coord[2], labels=paste(format(laps.times.to.goal,digits=1,nsmall=1), "min"), pos=4, col="blue", offset=2)
points(laps, laps.times.total-0.1, pch=24, col="orange")
grid()
plot(validation.data$distance, validation.data$speed, type="l", xlab="Distance [m]", ylab="Speed [m/s]", main="Speed")
lines(validation.data$distance, speed.prediction, col="magenta")
abline(v=laps, col="orange")
grid()
legend("topleft", legend=c("Real", "Estimated"), fill=c("black", "magenta"))
dev.off()



#--------------------------------------------------------------------------------------------------


