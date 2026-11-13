# Generate R reference values for the cross-language parity tests.
#
# Usage (from the sharedimp-py repository root; R is only needed to regenerate
# the committed reference, never to run the Python tests):
#
#   Rscript tests/fixtures/cross_language/generate_r_reference.R \
#       tests/fixtures/cross_language/inputs.json \
#       tests/fixtures/cross_language/r_reference.json \
#       /path/to/a/copy/of/sharedimp-r  <commit-of-that-copy>
#
# The R package is loaded from the given directory with pkgload::load_all().
# Pass a *copy* of the repository so that nothing is written into sharedimp-r.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
inp <- jsonlite::fromJSON(args[[1L]], simplifyVector = TRUE)
suppressMessages(pkgload::load_all(args[[3L]], quiet = TRUE))
suppressMessages(library(yardstick))

# I(): keep length-one vectors as JSON arrays.
num <- function(x) I(unname(as.numeric(x)))
chr <- function(x) I(unname(as.character(x)))

# ------------------------------------------------------------------ metrics
m <- inp$metrics
reg <- data.frame(truth = m$regression$truth, estimate = m$regression$estimate)
bin <- data.frame(
  truth = factor(m$binary$truth, levels = m$binary$levels),
  .pred_a = m$binary$prob_a, .pred_b = m$binary$prob_b
)
bin$class <- factor(m$binary$levels[max.col(as.matrix(bin[, 2:3]), ties.method = "first")],
  levels = m$binary$levels
)
mp <- as.matrix(m$multiclass$prob)
mul <- data.frame(
  truth = factor(m$multiclass$truth, levels = m$multiclass$levels),
  .pred_u = mp[, 1], .pred_v = mp[, 2], .pred_w = mp[, 3]
)
mul$class <- factor(m$multiclass$levels[max.col(mp, ties.method = "first")],
  levels = m$multiclass$levels
)

metrics <- list(
  regression = list(
    mse = mean((reg$truth - reg$estimate)^2),
    sharedimp_mse = as.numeric(sharedimp::mse(reg, truth, estimate)$.estimate),
    rmse = as.numeric(rmse(reg, truth, estimate)$.estimate),
    mae = as.numeric(mae(reg, truth, estimate)$.estimate),
    rsq = as.numeric(rsq(reg, truth, estimate)$.estimate),
    rsq_trad = as.numeric(rsq_trad(reg, truth, estimate)$.estimate)
  ),
  binary = list(
    mn_log_loss = as.numeric(mn_log_loss(bin, truth, .pred_a)$.estimate),
    accuracy = as.numeric(accuracy(bin, truth, class)$.estimate),
    roc_auc_event_first = as.numeric(roc_auc(bin, truth, .pred_a)$.estimate),
    roc_auc_event_second = as.numeric(roc_auc(bin, truth, .pred_b, event_level = "second")$.estimate)
  ),
  multiclass = list(
    mn_log_loss = as.numeric(mn_log_loss(mul, truth, .pred_u, .pred_v, .pred_w)$.estimate),
    accuracy = as.numeric(accuracy(mul, truth, class)$.estimate),
    roc_auc_hand_till = as.numeric(roc_auc(mul, truth, .pred_u, .pred_v, .pred_w)$.estimate)
  )
)

# ------------------------------------------------------- residualization
im <- inp$importance
X <- as.matrix(im$X)
colnames(X) <- im$feature_names
Xd <- as.matrix(im$X_degenerate)
colnames(Xd) <- im$degenerate_feature_names
folds <- im$folds
train <- which(folds != 1L)
assess <- which(folds == 1L)

resid_ref <- lapply(seq_len(ncol(Xd)), function(i) {
  rep <- build_representations(Xd[train, ], Xd[assess, ], anchor = i, drop_constant = TRUE)
  cf <- rep$coefs
  list(
    anchor = cf$anchor,
    features = chr(cf$features),
    slope = num(cf$slope),
    intercept = num(cf$intercept),
    dropped = chr(rep$dropped),
    full_columns = chr(colnames(rep$full_train)),
    reduced_columns = chr(colnames(rep$reduced_train)),
    full_train = unname(rep$full_train),
    full_assess = unname(rep$full_assess)
  )
})

# ------------------------------------------------------ shared_importance
run <- function(x, y, metric, name) {
  res <- suppressWarnings(
    shared_importance(x, y, learner = "linear", metric = metric, resamples = folds)
  )
  list(
    name = name,
    metric_direction = res$metric$direction,
    feature = chr(res$importance$feature),
    importance = num(res$importance$importance),
    sd = num(res$importance$sd),
    n_folds = num(res$importance$n_folds),
    fold_feature = chr(res$folds$feature),
    fold_id = chr(res$folds$id),
    fold_metric_full = num(res$folds$metric_full),
    fold_metric_reduced = num(res$folds$metric_reduced),
    fold_importance = num(res$folds$importance)
  )
}
y_reg <- im$y_regression
y_bin <- factor(im$y_binary, levels = c("a", "b"))
y_mul <- factor(im$y_multiclass, levels = c("u", "v", "w"))
x1 <- X[, 1, drop = FALSE]

importance <- list(
  regression = list(
    mse = run(X, y_reg, "mse", "mse"),
    rmse = run(X, y_reg, "rmse", "rmse"),
    mae = run(X, y_reg, "mae", "mae"),
    rsq = run(X, y_reg, "rsq", "rsq"),
    r2 = run(X, y_reg, yardstick::rsq_trad, "rsq_trad")
  ),
  regression_degenerate = list(mse = run(Xd, y_reg, "mse", "mse")),
  regression_single_feature = list(
    mse = run(x1, y_reg, "mse", "mse"),
    r2 = run(x1, y_reg, yardstick::rsq_trad, "rsq_trad")
  ),
  binary = list(
    log_loss = run(X, y_bin, "mn_log_loss", "mn_log_loss"),
    accuracy = run(X, y_bin, "accuracy", "accuracy"),
    roc_auc = run(X, y_bin, "roc_auc", "roc_auc")
  ),
  binary_single_feature = list(
    log_loss = run(x1, y_bin, "mn_log_loss", "mn_log_loss"),
    accuracy = run(x1, y_bin, "accuracy", "accuracy")
  ),
  multiclass = list(
    log_loss = run(X, y_mul, "mn_log_loss", "mn_log_loss"),
    accuracy = run(X, y_mul, "accuracy", "accuracy"),
    roc_auc = run(X, y_mul, "roc_auc", "roc_auc")
  )
)

out <- list(
  meta = list(
    generated = format(Sys.Date()),
    r_version = R.version.string,
    yardstick = as.character(packageVersion("yardstick")),
    parsnip = as.character(packageVersion("parsnip")),
    nnet = as.character(packageVersion("nnet")),
    sharedimp_r_version = as.character(packageVersion("sharedimp")),
    sharedimp_r_commit = args[[4L]],
    residualization_split = "training rows: folds != 1; assessment rows: folds == 1"
  ),
  metrics = metrics,
  residualization = resid_ref,
  importance = importance
)
writeLines(jsonlite::toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"), args[[2L]])
