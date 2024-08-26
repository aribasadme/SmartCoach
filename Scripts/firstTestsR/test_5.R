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

training.data <- do.call(rbind, datasets.all)
subset.data <- training.data[,c("slope", "speed")]
subset.data.filtered <- subset.data[((subset.data[,1] > slope.min) & (subset.data[,1] < slope.max) & (subset.data[,2] > speed.min) & (subset.data[,2] < speed.max)), ]
current.profile <- profile.2D(subset.data.filtered)

pdf("../runner_vs_stats.pdf", width=14, height=20, paper="special")
#par(mfrow=c(ceiling(sqrt(n.files)), floor(sqrt(n.files))), oma=c(0,0,0,0), mar=c(4,4,1,1))
par(mfrow=c(7,4), oma=c(0,0,0,0), mar=c(4,4,1,1))
for (run.out in 1:n.files) {
    validation.data <- datasets.all[[run.out]]
    
    distance.factor <- cut(validation.data$distance, seq(0, max(validation.data$distance)+distance.step, by=distance.step))
    speed.stats <- tapply(validation.data$speed, distance.factor, mean)
    slope.stats <- tapply(validation.data$slope, distance.factor, mean)
    
    prediction.list <- lapply(slope.stats, function(x){current.profile$stats[,((x > current.profile$intervals[1,])&(x <= current.profile$intervals[2,]))]})
    valid.prediction <- sapply(prediction.list, length) > 0
    prediction <- do.call(cbind, prediction.list[valid.prediction])
    
    plot(speed.stats[valid.prediction], type="s", ylim=c(speed.min, speed.max), lwd=1, xlab="Distance [hm]", ylab="Speed [m/s]")
    points(prediction[1,], type="s", col="green")
    points(prediction[2,], type="s", col="blue")
    points(prediction[3,], type="s", col="red")
    grid()
    legend("bottomright", legend=c("Runner", "Q 75%", "Q 50%", "Q25%"), fill=c("black", "red", "blue", "green"))
}
dev.off()




#--------------------------------------------------------------------------------------------------


