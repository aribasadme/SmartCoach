setwd("/Users/aRa/OneDrive/Documentos/ERASMUS/PFC/SmartCouch/smartCOACH/firstTestsR/data_corrected")

file.list <- list.files(pattern=".tab$")

file.name <- file.list[1]

dataset <- read.table(file.name, header=TRUE, sep="\t")
dataset.lenght <- nrow(dataset)

plot(dataset$distance, dataset$elevation, type="l")

position.change <- dataset$distance[-1] - dataset$distance[-dataset.lenght]
elevation.change <- dataset$elevation[-1] - dataset$elevation[-dataset.lenght]

slope <- elevation.change / position.change

plot(dataset$distance[-1], slope, type="l")

window.size.half <- 3
slope <- sapply(1:dataset.lenght, 
                function(x) {
                    index <- (x-window.size.half):(x+window.size.half)
                    index <- index[index > 0]
                    dataset.part <- dataset[index,]
                    model <- lm(elevation ~ distance, dataset.part)
                    coef(model)[2]
                })

par(mfrow=c(2,1), oma=c(0,0,0,0), mar=c(3,3,1,1))
plot(dataset$distance, dataset$elevation, type="l", xaxs="i")
grid()
plot(dataset$distance, slope, type="l", xaxs="i")
abline(h=0, col="red", lty=2)
grid()

test <- kmeans(cbind(slope, dataset$speed), 3, 100, 100)
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,1,1))
plot(slope, dataset$speed, pch=20, col=test$cluster, xlab="slope", ylab="speed")
points(test$centers, col=1:3)
grid()

library(rgl)
plot3d(slope, dataset$speed, dataset$distance)

#--------------------------------------------------------------------------------------------------
setwd("/home/hector/Desktop/work/APE_run/data")

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
    datasets.all[[i]] <- cbind(dataset, slope=slope)
    i <- i + 1
}

i <- 1
par(mfrow=c(2,1), oma=c(0,0,0,0), mar=c(4,4,1,1))
plot((datasets.all[[i]])$slope, (datasets.all[[i]])$speed, pch=20, ylim=c(0,6), xlab="slope", ylab="speed")
plot((datasets.all[[i]])$distance, (datasets.all[[i]])$elevation, type="l", xlab="distance", ylab="elevation")
grid()
i <- i + 1
#--------------------------------------------------------------------------------------------------


