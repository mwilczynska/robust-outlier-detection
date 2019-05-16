# Outliers

# Set working directory
setwd("/media/chawinski/data/Documents/work/InsightsandAnalyticsOfficer/r/outliers/images")

# Install required packages if not already installed
#install.packages(c("robustbase", "dbscan", "fpc", "factoextra", "randomForest)

# Load libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(robustbase)
library(truncnorm)
library(fpc)
library(randomForest)
set.seed(466) # Set particular random seed for reproducibility of results

# Clear global environment
rm(list = ls())

# Create a dataset
# rtruncnorm allows us to get a random set of numbers from a normal distribution 
x <- rtruncnorm(n=19, a=0, b=10, mean=5, sd=5) 
y <- x + rtruncnorm(n=19, a=-2.5, b=2.5, mean=5, sd=5)
x <- c(x, 2) # add 1st outlier
y <- c(y, 20) # add 1st outlier
x <- c(x, 2.5) # add 2nd outlier
y <- c(y, 11) # add 2nd outlier

# Create dataframe out of dataset
d <- data.frame(x, y)

# Fit standard linear regression
fit <- lm(y ~ x, data=d)

# Extract parameters from linear regression
d$predicted <- predict(fit) # Save the predicted values in a new column
d$residuals <- residuals(fit) # Save the residual values in a new column
d$zscore <- ((d$y - d$predicted)/sd(d$y - d$predicted)) # Save the zscore values in a new column

# Plot the data
ggplot(d, aes(x = x, y = y)) + #Set up canvas withoutcome variable on y axis
  geom_point() +# Plot the actual points
  theme_bw()  # Add theme for cleaner look

# Plot showing residual distances
ggplot(d, aes(x = x, y = y)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +  # Plot regression slope
  geom_segment(aes(xend = x, yend = predicted), alpha = 1.0, colour = 'red') +  # alpha to fade lines
  geom_point(aes(y = predicted), shape = 1) +
  geom_point() +
  theme_bw()  # Add theme for cleaner look

# Density plot can show information about the distribution shape of the residuals
# Deviations from a normal/Gaussian distribution may indicate outliers are present
ggplot(d, aes(x = residuals)) +
  geom_density(colour = "black", fill = "red") +
  theme_bw()  # Add theme for cleaner look 

# Simple box plot
ggplot(d, aes(x = x, y = y)) +
  stat_boxplot(geom = "errorbar") +
  geom_boxplot(outlier.colour = "red",  outlier.size = 4, fill='#A4A4A4') +
  theme_bw()

##############
# 3-sigma Rule
##############

# Clean plot with regression fit
ggplot(d, aes(x = x, y = y)) +
  geom_line(aes(y=d$predicted), size = 2, colour = "lightgrey") +
  geom_point() +
  theme_bw()  # Add theme for cleaner look

# zscore plot
ggplot(d, aes(x = x, y = zscore)) +
  geom_hline(yintercept=0, aes(colour = 'lightgrey', size = 2)) +
  geom_point(aes(colour = (d$zscore <= -3) | (d$zscore >= 3))) +
  scale_colour_manual(name = '|z-score| \u2265 3', values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

# Now remove outliers with zscore >= 3 and create a new dataframe without those outliers
e <- d[!(d$zscore >= 3), ]
drops <- c("predicted", "residuals", "zscore")
e <- e[ , !(names(e) %in% drops)]

# Fit a new linear regression to the data without the outliers
fit <- lm(e$y ~ e$x, data=e)
e$predicted <- predict(fit) # Save the predicted values
e$residuals <- residuals(fit) # save the residual values
e$zscore <- ((e$y - e$predicted)/sd(e$y - e$predicted))

# Plot the data with regression fit
ggplot(e, aes(x = x, y = y)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey", size = 2) +  # Plot regression slope
  geom_point() +
  theme_bw()  # Add theme for cleaner look


# Plot zscore a second time to check for outliers
ggplot(e, aes(x = x, y = zscore)) +
  geom_point(aes(colour = (e$zscore <= -3) | (e$zscore >= 3))) +
  scale_colour_manual(name = '|z-score| \u2265 3', values = setNames(c('red','black'),c(T, F))) +
  geom_hline(yintercept=0, aes(colour = 'black')) +
  theme_bw()  # Add theme for cleaner look


# Why sigma clipping can go wrong
# Clear global environment in preperation for a new dataset
rm(list = ls())
x <- rtruncnorm(n=50, a=0, b=20, mean=10, sd=2)
y <- 20+-1.5*x + rtruncnorm(n=50, a=5, b=10, mean=5, sd=1)
a <- rtruncnorm(n=25, a=10, b=20, mean=15, sd=3) # x values of outliers
b <- rtruncnorm(n=25, a=10, b=20, mean=17, sd=3) # y values of outliers
x <- append(x, a)
y <- append(y, b)
d <- data.frame(x, y) # Create dataframe with original data + outliers
fit <- lm(y ~ x, data=d) # Fit linear regression
d$predicted <- predict(fit) # Save the predicted values
d$residuals <- residuals(fit) # Save the residual values
d$zscore <- ((d$y - d$predicted)/sd(d$y - d$predicted)) # Save the z-score

# Plot data with linear regression
ggplot(d, aes(x = x, y = y)) +
  geom_line(aes(y=d$predicted), size = 2, colour = "lightgrey") +
  geom_point() +
  theme_bw()  # Add theme for cleaner look

# zscore plot
ggplot(d, aes(x = x, y = zscore)) +
  geom_hline(yintercept=0, colour = "lightgrey", size = 2) +
  geom_point(aes(colour = (d$zscore <= -3) | (d$zscore >= 3))) +
  scale_colour_manual(name = '|z-score| \u2265 3', values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

# This plot produces no outliers! Sigma clipping willnot work since there is nothing to clip
# The standard deviation is hugely inflated
cat("Standard Deviation of OLS fit: ", sd(residuals(fit)))

####################################
# Least Trimmed Squares (LTS) method
####################################

# Calculate Least Trimmed Squares (LTS) fit
lts <- ltsReg(y~x, alpha = 0.65) # Calculate LTS fit with 65% of the original data
d$ltspredicted <- fitted.values(lts) # Save the predicted values
d$ltsresiduals <- residuals(lts) # Save the residual values
sd <- sd(d$x[lts[["best"]]]) # Save the standard deviation
d$ltszscore <- lts[["residuals"]]/sd # Save the zscore

# Clean data with regression fit
ggplot(d, aes(x = x, y = y)) +
  geom_line(aes(y=d$ltspredicted), size = 2, colour = "lightgrey") +
  geom_point() +
  theme_bw()  # Add theme for cleaner look

# zscore fit
ggplot(d, aes(x = x, y = ltszscore)) +
  geom_hline(yintercept=0, aes(colour = "lightgrey", size = 2)) +
  geom_point(aes(colour = (d$ltszscore <= -3) | (d$ltszscore >= 3))) +
  labs(y = "residuals") +
  scale_colour_manual(name = '|LTS residual| \u2265 3', values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

# Remove outliers
e <- d[!(d$ltszscore >= 3), ] # Create new dataframe without outliers
drops <- c("ltspredicted", "ltsresiduals", "ltszscore")
e <- e[ , !(names(e) %in% drops)]

# Calculate new LTS fit
lts <- ltsReg(e$y~e$x, alpha = 0.65) # Calculate LTS fit with 65% of the original data
e$ltspredicted <- fitted.values(lts) # Save the predicted values
e$ltsresiduals <- residuals(lts) # Save the residual values
ltsi <- lts[["coefficients"]][1] # y-intercept of LTS fit
ltsg <- lts[["coefficients"]][2] # slope of LTS fit
sd <- sd(e$x[lts[["best"]]]) # Standard deviation of LTS fit
e$ltszscore <- lts[["residuals"]]/sd # zscore of LTS fit

# Plot LTS fit without outliers
ggplot(e, aes(x = x, y = y)) +
  geom_line(aes(y=e$ltspredicted), size = 2, colour = "lightgrey") +
  geom_point() +
  theme_bw()  # Add theme for cleaner look

# zscore fit - check if there are any new outliers 
ggplot(e, aes(x = x, y = ltszscore)) +
  geom_hline(yintercept=0, aes(colour = "lightgrey", size = 2)) +
  geom_point(aes(colour = (e$ltszscore <= -3) | (e$ltszscore >= 3))) +
  labs(y = "residuals") +
  scale_colour_manual(name = '|LTS residuals| \u2265 3', values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

#################
# Cook's distance
#################

# Calculate Cooke's distance
d$cd <- cooks.distance(fit)
# Remove outliers
f <- d[!(d$cd > 4/nrow(d)), ]
drops <- c("ltspredicted", "ltsresiduals", "ltszscore")
f <- f[ , !(names(f) %in% drops)]

# Run new linear regression without outliers
fitcd <- lm(y ~ x, data=f)
cdi <- fitcd[["coefficients"]][1]
cdg <- fitcd[["coefficients"]][2]

# Plot individual Cooke's distances
plot(fitcd, which=4, cook.levels=d$cd > 4.5/nrow(d))

# Plot Cook'es distance fit with outliers removed
ggplot(d, aes(x = x, y = y)) +
  geom_abline(intercept = cdi, slope=cdg, colour = "lightgrey", size = 2) +
  geom_point() +
  geom_point(aes(colour = d$cd > 4.5/nrow(d))) +
  scale_colour_manual(name = "Cook's Distance Outlier", values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

########
# DBSCAN
########

# Calculate clusters and outliers by DBSCAN
db <- fpc::dbscan(d[, -c(9, 8, 7, 6, 5, 4, 3)], eps = 2.9, MinPts = 3)
d$db <- db[["cluster"]]

# Visualisation of DBSCAN clusters
plot(db, d[, -c(10, 9, 8, 7, 6, 5, 4, 3)], main = "DBSCAN", frame = FALSE)

# Removing outliers and second cluster
g <- d[(d$db == 1), ]
fitdb <- lm(y ~ x, data = g) # Linear regression on main cluster
dbi <- fitdb[["coefficients"]][1] # Intercept
dbg <- fitdb[["coefficients"]][2] # Slope

# Plot linear regression with DBSCAN ouliers removed
ggplot(d, aes(x = x, y = y)) +
  geom_abline(intercept = dbi, slope=dbg, colour = "lightgrey", size = 2) +
  geom_point() +
  geom_point(aes(colour = d$db != 1)) +
  scale_colour_manual(name = "DBSCAN Outlier", values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

###############
# Random forest
###############

# Calculate random forest
h <- d[, c("x", "y")] # Extract only x and y values to a new dataframe
fitrf <- randomForest(h,
                  ntree=500, # Number of trees in random forest
                  importance=T, 
                  proximity=T)

# Add proximity variable to main dataframe
d$proximity <- outlier(fitrf$proximity)

# Remove outliers with proximity >= |0.5|
h$proximity <- outlier(fitrf$proximity)
prox <- 0.5
h <- h[!((h$proximity <= -prox) | (h$proximity >= prox)), ]

# Linear regression without outliers
lrrf <- lm(y ~ x, data = h)
rfi <- lrrf[["coefficients"]][1] # Intercept
rfs <- lrrf[["coefficients"]][2] # Slope

# Plot proximity
ggplot(d, aes(x = x, y = proximity)) +
  geom_hline(yintercept=0, colour = "lightgrey", size = 2) +
  geom_point(aes(colour = (d$proximity <= -prox) | (d$proximity >= prox))) +
  scale_colour_manual(name = '|proximity| \u2265 0.5', values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

# Plot linear regression with random forest outliers removed
ggplot(d, aes(x = x, y = y)) +
  geom_abline(intercept = rfi, slope=rfs, colour = "lightgrey", size = 2) +
  geom_point() +
  geom_point(aes(colour = (d$proximity <= -prox) | (d$proximity >= prox))) +
  scale_colour_manual(name = "RF Outlier", values = setNames(c('red','black'),c(T, F))) +
  theme_bw()  # Add theme for cleaner look

# Compare all methods
ggplot(d, aes(x = x, y = y)) +
  geom_abline(aes(intercept = rfi, slope=rfs, colour = "Random Forest"), size = 1, show.legend=TRUE) +
  geom_abline(aes(intercept = dbi, slope=dbg, colour = "DBSCAN"), size = 1, show.legend=TRUE) +
  geom_abline(aes(intercept = cdi, slope=cdg, colour = "Cook's Distance"), size = 1, show.legend=TRUE) +
  geom_abline(aes(intercept = ltsi, slope=ltsg, colour = "LTS"), size = 1, show.legend=TRUE) +
  geom_smooth(method = "lm", se = FALSE, aes(color = "All data"), size = 1, show.legend=TRUE) +  # Plot regression slope
  ggtitle("Outlier detection methods") +
  geom_point() +
  theme_bw()  # Add theme for cleaner look


#####################
# Non-parametric data
#####################

# Load the data from an existing package
data("multishapes", package = "factoextra")
df <- multishapes[, 1:2]

# Compute DBSCAN using fpc package
library("fpc")

# Calculate DBSCAN clusters and outliers
db2 <- fpc::dbscan(df, eps = 0.15, MinPts = 5)

# Plot DBSCAN results
library("factoextra")
fviz_cluster(db2, data = df, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE, 
             geom = "point",palette = "jco", ggtheme = theme_classic())

###############################
# Extra randomly generated data
###############################
rm(list = ls())
x = c(-81,  -16 , -70 , -95,  -64 ,  84 ,  -3,   17,  -15 , -34 , -42,   17 , -23, -170, -117,  -63, -120,  -14,    0,  -34,  -12 , 107,-7,  -59 ,  59,   17,  170,  -64,  -96,  -60 , -84,   27 ,  66 ,  63,   55 ,  37 ,  11  ,-15)
y = c( -54,  -12,    0,  -24 , -66,    0 , -60,    6,    0  ,  0 , -24  ,  6,    0, -126  ,  6,    0 , -78 , -12,    0,    0 , -36,    0,   0,    0,    0,    0,    0,    0,    0,    6,    0,    6,   12,   -6,    0,   42,    6,    6)
plot(x,y)
reg = lm(y ~ x+ 0)
abline(reg,col ='black')
reg_lts <- ltsReg(y~x, alpha = 0.85)
abline(reg_lts,col ='red')
A = cbind(x,y,reg$residuals)
## Remove the two observation with highest residuals, two since floor(38*0.95) = 36
y = y[c(-7,-14)]
x = x[c(-7,-14)]
reg2 = lm(y ~ x+ 0)
abline(reg2,col ='blue')

# another LTS example
rm(list = ls())
set.seed(466)
n <- 100
x9 <- rnorm(n)
y9 <- 1 - 2*x9 + rnorm(n)
r0 <- lm(y9~x9)
y9[ sample(1:n, floor(n/4)) ] <- 7
plot(y9~x9)
#points(y9~x9)
r1 <- lm(y9~x9)
r2 <- ltsReg(y9~x9, alpha = 0.85)
abline(r0, col = 'black', lty = 2)
abline(r1, col='black')
abline(r2, col='red')
#abline(1,-2,lty=3)
legend( par("usr")[1], par("usr")[3], yjust=0,
        c("Linear regression", "Trimmed regression"),
        lty=1, lwd=1,
        col=c("black", "red") )
title("Least Trimmed Squares (LTS)")


# Another set of outliers
rm(list = ls())
set.seed(466)
n <- 100
x9 <- rnorm(n)
y9 <- 1 - 2*x9 + rnorm(n)
r0 <- lm(y9~x9)
a1 <- rnorm(0.5*n, mean=1, sd=0.5)
a2 <- rnorm(0.5*n, mean=2, sd=0.25)^2
x9 <- append(x9, a1)
y9 <- append(y9, a2)
plot(y9~x9)
r1 <- lm(y9~x9)
r2 <- ltsReg(y9~x9, alpha = 0.50)
abline(r0, col = 'black', lty = 2)
abline(r1, col='black')
abline(r2, col='red')
#abline(1,-2,lty=3)
legend( par("usr")[1], par("usr")[3], yjust=0,
        c("Linear regression", main= expression(paste("Least Trimmed Squares, ", alpha, "=0.85    "))),
        lty=1, lwd=1,
        col=c("red", "black") )
title("Least Trimmed Squares (LTS)")

## End ##
