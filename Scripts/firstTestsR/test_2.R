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
    datasets.all[[i]] <- cbind(dataset, slope=slope)
    i <- i + 1
}

all.data <- do.call(rbind, datasets.all)
linear.model <- lm(speed ~ slope, all.data)

pdf("andresSlopeSpeed.pdf", width=6, height=6, paper="special")
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,2,2))
plot(all.data$slope, all.data$speed, pch=".", xlim=c(-0.4, 0.4), ylim=c(0,5), xlab="Slope [%]", ylab="Speed [m/s]", main="Andres pace")
abline(a=coef(linear.model)[1], b=coef(linear.model)[2], col="blue")
grid()
dev.off()

slope.min <- -0.2
slope.max <- 0.2
speed.min <- 0
speed.max <- 5
slope.resolution <- 25
certainty.threshold <- 100
all.data.filtered <- all.data[((all.data$slope > slope.min)&(all.data$slope < slope.max)&(all.data$speed > speed.min)&(all.data$speed < speed.max)), ]

linear.model <- lm(speed ~ slope, all.data.filtered)
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,1,1))
plot(all.data.filtered$slope, all.data.filtered$speed, pch=".", xlab="slope", ylab="speed")
grid()
abline(a=coef(linear.model)[1], b=coef(linear.model)[2], col="red")

slope.factor <- cut(all.data.filtered$slope, slope.resolution)
slope.intervals <- sapply(strsplit(levels(slope.factor), ","), function(x){as.numeric(c(sub("[(]", "", x[1]), sub("[]]", "", x[2])))})
slope.mids <- apply(slope.intervals, 2, function(x){mean(x)})
slope.counts <- as.numeric(table(slope.factor))
data.split <- split(all.data.filtered$speed, slope.factor)
q.123 <- sapply(data.split, quantile, probs=c(0.25, 0.5, 0.75))

certainty <- slope.counts / certainty.threshold
certainty[certainty>1] <- 1

pdf("../andresPaceSlope.pdf", width=8, height=6, paper="special")
par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,2,1))
plot(slope.mids, q.123[2,], type="l", ylim=c(2, 4), xlab="Slope [%]", ylab="Speed [m/s]", main="Andres pace")
rect(slope.intervals[1,], speed.min, slope.intervals[2,], speed.max, density=25*(1-certainty), angle=90, border=NA)
lines(slope.mids, q.123[1,], col="green")
lines(slope.mids, q.123[3,], col="red")
lines(slope.mids, q.123[2,], lwd=2, col="blue")
legend("bottom", legend=c("Q 75%", "Q 50%", "Q 25%"), fill=c("red", "blue", "green"))
grid()
dev.off()


par(mfrow=c(1,1), oma=c(0,0,0,0), mar=c(4,4,1,1))
plot(slope.mids, q.123[2,], type="l", ylim=c(min(all.data.filtered$speed, na.rm=TRUE), max(all.data.filtered$speed, na.rm=TRUE)), col="red")
rect(slope.intervals[1,], q.123[1,], slope.intervals[2,], q.123[3,], col=gray(certainty), border=NA)
lines(slope.mids, q.123[1,], col="blue")
lines(slope.mids, q.123[3,], col="blue")
lines(slope.mids, q.123[2,], col="red")
grid()
#--------------------------------------------------------------------------------------------------


