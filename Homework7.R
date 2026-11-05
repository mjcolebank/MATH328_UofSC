# Homework 7: California Housing Regression in R
# Expects california_housing_small.csv in the working directory.

set.seed(223)
housing <- read.csv("california_housing_small.csv")

X_all <- as.matrix(housing[, c("AveBedrms", "AveOccup", "AveRooms",
                               "HouseAge", "MedInc")])
y <- housing$MedHouseVal
X_names <- c("Average Bedrooms", "Average Occupancy", "Average # Rooms",
             "Average House Age", "Median Household Income")
y_name <- "Median House Value"

# Part A: descriptive statistics and pairwise plots
mean_X <- colMeans(X_all)
var_X <- apply(X_all, 2, var)
cat("X variables:\n\n")
for (i in seq_len(ncol(X_all))) {
  cat(sprintf("Variable: %s, Mean: %.6f, Variance: %.6f\n",
              X_names[i], mean_X[i], var_X[i]))
}
cat(sprintf("\nY variable: %s, Mean: %.6f, Variance: %.6f\n",
            y_name, mean(y), var(y)))

par(mfrow = c(2, 3))
for (i in seq_len(ncol(X_all))) {
  plot(X_all[, i], y, pch = 16, col = rgb(1, 0, 0, 0.3),
       xlab = X_names[i], ylab = y_name, main = X_names[i])
  grid()
}
par(mfrow = c(1, 1))

# Shared helpers and train/test split
x_income <- X_all[, 5]
ids <- sample(seq_len(length(y)))
n_train <- floor(0.8 * length(y))
id_train <- ids[seq_len(n_train)]
id_test <- ids[(n_train + 1):length(y)]

sse <- function(y, yhat) sum((y - yhat)^2)
r2 <- function(y, yhat) {
  1 - sse(y, yhat) / sum((y - mean(y))^2)
}
fit_least_squares <- function(X, y) qr.solve(X, y)
make_polynomial_matrix <- function(x, order) {
  do.call(cbind, lapply(0:order, function(power) x^power))
}

# Part B: polynomial models for median income
par(mfrow = c(2, 3))
for (i in seq_len(ncol(X_all))) {
  plot(X_all[id_train, i], y[id_train], pch = 16,
       col = rgb(1, 0, 0, 0.3), xlab = X_names[i],
       ylab = y_name, main = paste(X_names[i], "train/test"))
  points(X_all[id_test, i], y[id_test], pch = 16,
         col = rgb(0, 0, 1, 0.3))
  grid()
}
par(mfrow = c(1, 1))

orders <- 1:5
results <- data.frame()
models <- list()
R2_train <- numeric(length(orders))
R2_test <- numeric(length(orders))

for (order in orders) {
  X_train <- make_polynomial_matrix(x_income[id_train], order)
  X_test <- make_polynomial_matrix(x_income[id_test], order)
  theta <- fit_least_squares(X_train, y[id_train])
  yhat_train <- as.vector(X_train %*% theta)
  yhat_test <- as.vector(X_test %*% theta)
  models[[as.character(order)]] <- theta
  R2_train[order] <- r2(y[id_train], yhat_train)
  R2_test[order] <- r2(y[id_test], yhat_test)
  results <- rbind(results, data.frame(
    order = order, parameters = length(theta),
    train_SSE = sse(y[id_train], yhat_train),
    test_SSE = sse(y[id_test], yhat_test),
    train_R2 = R2_train[order], test_R2 = R2_test[order]
  ))
}
print(results)

income_plot <- seq(min(x_income), max(x_income), length.out = 250)
plot(x_income[id_train], y[id_train], pch = 16,
     col = rgb(1, 0, 0, 0.4), xlab = "MedInc",
     ylab = y_name)
points(x_income[id_test], y[id_test], pch = 16,
       col = rgb(0, 0, 1, 0.4))
for (order in orders) {
  X_plot <- make_polynomial_matrix(income_plot, order)
  lines(income_plot, X_plot %*% models[[as.character(order)]],
        col = order + 1, lwd = 2)
}
legend("topleft", legend = c("training", "testing", paste("order", orders)),
       col = c("red", "blue", 2:6), pch = c(16, 16, rep(NA, 5)),
       lty = c(NA, NA, rep(1, 5)), bty = "n")

# Part B: AIC model selection
n <- length(id_train)
results$AIC <- n * log(results$train_SSE / n) + 2 * results$parameters
print(results[, c("order", "parameters", "train_SSE", "test_SSE",
                  "train_R2", "test_R2", "AIC")])
cat("Best order by AIC:", results$order[which.min(results$AIC)], "\n")
cat("Best order by testing SSE:", results$order[which.min(results$test_SSE)], "\n")

# Part C: multivariate linear regression
X_m1 <- cbind(1, X_all[, 5])
X_m2 <- cbind(1, X_all[, 3], X_all[, 5])
X_m3 <- cbind(1, X_all)

cat("Sizes of the data matrices:\n")
print(dim(X_m1)); print(dim(X_m2)); print(dim(X_m3))

theta_m1 <- fit_least_squares(X_m1[id_train, , drop = FALSE], y[id_train])
theta_m2 <- fit_least_squares(X_m2[id_train, , drop = FALSE], y[id_train])
theta_m3 <- fit_least_squares(X_m3[id_train, , drop = FALSE], y[id_train])

cat("Model 1 parameters:"); print(theta_m1)
cat("Model 2 parameters:"); print(theta_m2)
cat("Model 3 parameters:"); print(theta_m3)

predict_model <- function(X, theta, ids) {
  as.vector(X[ids, , drop = FALSE] %*% theta)
}
yhat_train_m1 <- predict_model(X_m1, theta_m1, id_train)
yhat_test_m1 <- predict_model(X_m1, theta_m1, id_test)
yhat_train_m2 <- predict_model(X_m2, theta_m2, id_train)
yhat_test_m2 <- predict_model(X_m2, theta_m2, id_test)
yhat_train_m3 <- predict_model(X_m3, theta_m3, id_train)
yhat_test_m3 <- predict_model(X_m3, theta_m3, id_test)

train_sse <- c(sse(y[id_train], yhat_train_m1),
               sse(y[id_train], yhat_train_m2),
               sse(y[id_train], yhat_train_m3))
test_sse <- c(sse(y[id_test], yhat_test_m1),
              sse(y[id_test], yhat_test_m2),
              sse(y[id_test], yhat_test_m3))
train_r2 <- c(r2(y[id_train], yhat_train_m1),
              r2(y[id_train], yhat_train_m2),
              r2(y[id_train], yhat_train_m3))
test_r2 <- c(r2(y[id_test], yhat_test_m1),
             r2(y[id_test], yhat_test_m2),
             r2(y[id_test], yhat_test_m3))

model_results <- data.frame(
  model = c("income", "# rooms + income", "all predictors"),
  train_SSE = train_sse, test_SSE = test_sse,
  train_R2 = train_r2, test_R2 = test_r2,
  AIC = n * log(train_sse / n) + 2 * c(length(theta_m1),
                                        length(theta_m2),
                                        length(theta_m3))
)
print(model_results)
