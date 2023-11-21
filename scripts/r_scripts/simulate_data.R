regression <- function(seed, N0){
    set.seed(seed)
    dat <- data.frame(array(rnorm(N0*8),dim=c(N0,8)))
    dat$X8 <- 1/2 * dat$X2 + sqrt(1-(1/2)^2)*dat$X8
    dat$z <- dat$X1/2 - dat$X2^2/4 + abs(dat$X3) * sin(2*dat$X3)/2 + dat$X4*dat$X5/2 + dat$X5^2*dat$X6/8
    dat$y <- rnorm(n=N0, mean=dat$z, sd=1)
    dat
}

# Saving dataframes to .csv files
write.csv(regression(100, N), file.path(output_dir, "train_data.csv"), row.names = FALSE)
write.csv(regression(200, N), file.path(output_dir, "test_data.csv"), row.names = FALSE)
