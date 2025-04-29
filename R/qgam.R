library(qgam); library(MASS)
if( suppressWarnings(require(RhpcBLASctl)) ){ blas_set_num_threads(1) } # Optional

fit <- qgam(accel~s(times, k=20, bs="ad"), 
            data = mcycle, 
            qu = 0.8)

# Plot the fit
xSeq <- data.frame(cbind("accel" = rep(0, 1e3), "times" = seq(2, 58, length.out = 1e3)))
pred <- predict(fit, newdata = xSeq, se=TRUE)
results <- append(pred, xSeq)
# save it
write.csv(as.data.frame(results), "results/realdata/qgam_acceleration.csv", row.names =FALSE)

plot(mcycle$times, mcycle$accel, xlab = "Times", ylab = "Acceleration", ylim = c(-150, 80))
lines(xSeq$times, pred$fit, lwd = 1)
lines(xSeq$times, pred$fit + 2*pred$se.fit, lwd = 1, col = 2)
lines(xSeq$times, pred$fit - 2*pred$se.fit, lwd = 1, col = 2)

# qGAM automatically fits the learning rate
check(fit$calibr, 2)

# Heteroscedastic Data
set.seed(651)
n <- 2000
x <- seq(-4, 3, length.out = n)
X <- cbind(1, x, x^2)
beta <- c(0, 1, 1)
sigma =  1.2 + sin(2*x)
f <- drop(X %*% beta)
dat <- f + rnorm(n, 0, sigma)
dataf <- data.frame(cbind(dat, x))
names(dataf) <- c("y", "x")

qus <- seq(0.05, 0.95, length.out = 5)
plot(x, dat, col = "grey", ylab = "y")
for(iq in qus){ lines(x, qnorm(iq, f, sigma)) }



# fit more models and compare to ground truth!
fit <- mqgam(y~s(x, k = 30, bs = "cr"), 
             data = dataf,
             qu = qus)

qus <- seq(0.05, 0.95, length.out = 5)
plot(x, dat, col = "grey", ylab = "y")
for(iq in qus){ 
  lines(x, qnorm(iq, f, sigma), col = 2)
  lines(x, qdo(fit, iq, predict))
}
legend("top", c("truth", "fitted"), col = 2:1, lty = rep(1, 2))



## We can get better credible intervals, and solve the "wigglines" problem for the top quantile, by letting the learning rate vary with the covariate. In particular, we can use an additive model for quantile location and one for learning rate.

fit <- qgam(list(y~s(x, k = 30, bs = "cr"), ~ s(x, k = 30, bs = "cr")), 
            data = dataf, qu = 0.95)
## Estimating learning rate. Each dot corresponds to a loss evaluation. 
## qu = 0.95.........done
plot(x, dat, col = "grey", ylab = "y")
tmp <- predict(fit, se = TRUE)
lines(x, tmp$fit)
lines(x, tmp$fit + 3 * tmp$se.fit, col = 2)
lines(x, tmp$fit - 3 * tmp$se.fit, col = 2)



fit <- qgam(list(accel ~ s(times, k=20, bs="ad"), ~ s(times)),
            data = mcycle, 
            qu = 0.8)
## Estimating learning rate. Each dot corresponds to a loss evaluation. 
## qu = 0.8................done
pred <- predict(fit, newdata = xSeq, se=TRUE)
plot(mcycle$times, mcycle$accel, xlab = "Times", ylab = "Acceleration", ylim = c(-150, 80))
lines(xSeq$times, pred$fit, lwd = 1)
lines(xSeq$times, pred$fit + 2*pred$se.fit, lwd = 1, col = 2)
lines(xSeq$times, pred$fit - 2*pred$se.fit, lwd = 1, col = 2)  
