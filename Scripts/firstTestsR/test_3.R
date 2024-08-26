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

speed.min <- 0
speed.max <- 5
distance.resolution <- 25
certainty.threshold <- 100
all.data.filtered <- all.data[((all.data$speed > speed.min)&(all.data$speed < speed.max)), ]

distance.factor <- cut(all.data.filtered$distance, distance.resolution)
distance.intervals <- sapply(strsplit(levels(distance.factor), ","), function(x){as.numeric(c(sub("[(]", "", x[1]), sub("[]]", "", x[2])))})
distance.mids <- apply(distance.intervals, 2, function(x){mean(x)})
distance.counts <- as.numeric(table(distance.factor))
data.split <- split(all.data.filtered$speed, distance.factor)
q.123 <- sapply(data.split, quantile, probs=c(0.25, 0.5, 0.75))

certainty <- distance.counts / certainty.threshold
certainty[certainty>1] <- 1

pdf("../andresPaceDistance.pdf", width=8, height=6, paper="special")
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,2,1))
plot(distance.mids, q.123[2,], type="l", ylim=c(2, 4), xlab="Distance [%]", ylab="Speed [m/s]", main="Andres pace")
rect(distance.intervals[1,], speed.min, distance.intervals[2,], speed.max, density=25*(1-certainty), angle=90, border=NA)
lines(distance.mids, q.123[1,], col="green")
lines(distance.mids, q.123[3,], col="red")
lines(distance.mids, q.123[2,], lwd=2, col="blue")
legend("bottom", legend=c("Q 75%", "Q 50%", "Q 25%"), fill=c("red", "blue", "green"))
grid()
dev.off()

#--------------------------------------------------------------------------------------------------


