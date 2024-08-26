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
    dataset$distance <- dataset$distance / max(dataset$distance, na.rm=TRUE)
    datasets.all[[i]] <- cbind(dataset, slope=slope)
    i <- i + 1
}

all.data <- do.call(rbind, datasets.all)

subset.data <- all.data[,c("distance", "elevation")]
v.x.min <- 0                                #distance
v.x.max <- 1                                #distance
v.y.min <- 0                                #elevation
v.y.max <- max(subset.data[,2], na.rm=TRUE) #elevation
v.x.resolution <- 25
certainty.threshold <- 100

subset.data.filtered <- subset.data[((subset.data[,1] > v.x.min) & (subset.data[,1] < v.x.max) & (subset.data[,2] > v.y.min) & (subset.data[,2] < v.y.max)), ]

v.x.factor <- cut(subset.data.filtered[,1], v.x.resolution)
v.x.intervals <- sapply(strsplit(levels(v.x.factor), ","), function(x){as.numeric(c(sub("[(]", "", x[1]), sub("[]]", "", x[2])))})
v.x.mids <- apply(v.x.intervals, 2, function(x){mean(x)})
v.x.counts <- as.numeric(table(v.x.factor))
data.split <- split(subset.data.filtered[,2], v.x.factor)
q.123 <- sapply(data.split, quantile, probs=c(0.25, 0.5, 0.75))

certainty <- v.x.counts / certainty.threshold
certainty[certainty>1] <- 1

pdf("../andresElevationDistance.pdf", width=6, height=6, paper="special")
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,2,1))
plot(v.x.mids, q.123[2,], type="l", ylim=c(v.y.min, v.y.max), xlab="Distance [%]", ylab="Elevation [m]", main="Andres pace")
rect(v.x.intervals[1,], v.x.min, v.x.intervals[2,], v.y.max, density=25*(1-certainty), angle=90, border=NA)
lines(v.x.mids, q.123[1,], col="green")
lines(v.x.mids, q.123[3,], col="red")
lines(v.x.mids, q.123[2,], lwd=2, col="blue")
legend("bottom", legend=c("Q 75%", "Q 50%", "Q 25%"), fill=c("red", "blue", "green"))
grid()
dev.off()

#--------------------------------------------------------------------------------------------------


