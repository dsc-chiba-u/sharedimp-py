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
# Pass a *copy* of the repository (e.g. `git archive HEAD | tar -x -C <dir>`)
# so that nothing is written into sharedimp-r and exactly the committed state
# is used.
#
# The output follows the common output schema of the specification as the R
# package reports it; nothing is converted or reordered here.

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
inp <- jsonlite::fromJSON(args[[1L]], simplifyVector = TRUE)
suppressMessages(pkgload::load_all(args[[3L]], quiet = TRUE))
suppressMessages(library(yardstick))

# I(): keep length-one vectors as JSON arrays.
num <- function(x) I(unname(as.numeric(x)))
chr <- function(x) I(unname(as.character(x)))
lgl <- function(x) I(unname(as.logical(x)))
mat <- function(x) unname(as.matrix(x))

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

# ------------------------------------------------------------------- designs
im <- inp$importance
named <- function(x, names) {
  x <- as.matrix(x)
  colnames(x) <- names
  x
}
X <- named(im$X, im$feature_names)
Xd <- named(im$X_degenerate, im$degenerate_feature_names)
Xc <- named(im$X_constant, im$constant_feature_names)
folds <- im$folds
y_reg <- im$y_regression
y_bin <- factor(im$y_binary, levels = c("a", "b"))
y_mul <- factor(im$y_multiclass, levels = c("u", "v", "w"))
x1 <- X[, 1, drop = FALSE]

sf <- inp$spec_fixture
Xs <- named(sf$X, sf$feature_names)
Xsc <- named(sf$X_collinear, sf$collinear_feature_names)

# ------------------------------- residualization, representations, predictions
# Training rows: folds != 1; assessment rows: folds == 1. Predictions are those
# of the built-in "linear" learner (lm.fit, aliased coefficients set to 0).
train <- which(folds != 1L)
assess <- which(folds == 1L)
ols <- as_learner("linear", "regression", NULL)

representation_reference <- function(x) {
  lapply(seq_len(ncol(x)), function(i) {
    constant_anchor <- FALSE
    rep <- withCallingHandlers(
      build_representations(x[train, , drop = FALSE], x[assess, , drop = FALSE], anchor = i),
      sharedimp_constant_anchor = function(w) {
        constant_anchor <<- TRUE
        invokeRestart("muffleWarning")
      }
    )
    cf <- rep$coefs
    stopifnot(identical(cf$constant_anchor, constant_anchor))
    list(
      anchor = cf$anchor,
      features = chr(cf$features),
      slope = num(cf$slope),
      intercept = num(cf$intercept),
      collinear = lgl(cf$collinear),
      constant_anchor = cf$constant_anchor,
      full_columns = chr(colnames(rep$full_train)),
      reduced_columns = chr(colnames(rep$reduced_train)),
      full_train = mat(rep$full_train),
      full_assess = mat(rep$full_assess),
      reduced_train = mat(rep$reduced_train),
      reduced_assess = mat(rep$reduced_assess),
      pred_full = num(ols$predict(ols$fit(rep$full_train, y_reg[train]), rep$full_assess)),
      pred_reduced = num(ols$predict(ols$fit(rep$reduced_train, y_reg[train]), rep$reduced_assess))
    )
  })
}

null_reg <- null_learner("regression", NULL)
null_bin <- null_learner("classification", levels(y_bin))
null_model <- list(
  regression_prediction = num(null_reg$predict(
    null_reg$fit(x1[train, , drop = FALSE][, 0, drop = FALSE], y_reg[train]),
    x1[assess, , drop = FALSE][, 0, drop = FALSE]
  )),
  binary_levels = chr(levels(y_bin)),
  binary_probabilities = mat(null_bin$predict(
    null_bin$fit(x1[train, , drop = FALSE][, 0, drop = FALSE], y_bin[train]),
    x1[assess, , drop = FALSE][, 0, drop = FALSE]
  ))
)

# ------------------------------------------------------- shared_importance
# Both result tables exactly as returned by the R package.
tables <- function(res) {
  imp <- res$importance
  fr <- fold_results(res)
  list(
    metric = res$metric$name,
    summary = list(
      feature = chr(imp$feature),
      importance = num(imp$importance),
      importance_std = num(imp$importance_std),
      score_full = num(imp$score_full),
      score_reduced = num(imp$score_reduced),
      n_splits = num(imp$n_splits)
    ),
    folds = list(
      feature = chr(fr$feature),
      split = num(fr$split),
      n_train = num(fr$n_train),
      n_test = num(fr$n_test),
      score_full = num(fr$score_full),
      score_reduced = num(fr$score_reduced),
      importance = num(fr$importance),
      failed = lgl(!is.na(fr$error)),
      error = chr(ifelse(is.na(fr$error), "", fr$error))
    )
  )
}

# Comparisons use the aborting mode, so that a failure cannot be mistaken for a
# value; the learner-failure cases below use the default ("warn").
run <- function(x, y, ..., resamples = folds, on_error = "error", learner = "linear") {
  n_warn <- 0L
  res <- withCallingHandlers(
    shared_importance(x, y, learner = learner, ..., resamples = resamples, on_error = on_error),
    warning = function(w) {
      n_warn <<- n_warn + 1L
      invokeRestart("muffleWarning")
    }
  )
  c(tables(res), list(n_warnings = n_warn))
}

importance <- list(
  regression = list(
    default = run(X, y_reg),
    mse = run(X, y_reg, metric = "mse"),
    rmse = run(X, y_reg, metric = "rmse"),
    mae = run(X, y_reg, metric = "mae"),
    r2 = run(X, y_reg, metric = "r2"),
    rsq = run(X, y_reg, metric = "rsq")
  ),
  regression_degenerate = list(mse = run(Xd, y_reg, metric = "mse")),
  regression_constant = list(mse = run(Xc, y_reg, metric = "mse", on_error = "warn")),
  regression_single_feature = list(
    default = run(x1, y_reg),
    r2 = run(x1, y_reg, metric = "r2")
  ),
  binary = list(
    default = run(X, y_bin),
    log_loss = run(X, y_bin, metric = "log_loss"),
    accuracy = run(X, y_bin, metric = "accuracy"),
    roc_auc = run(X, y_bin, metric = "roc_auc")
  ),
  binary_single_feature = list(
    default = run(x1, y_bin),
    accuracy = run(x1, y_bin, metric = "accuracy")
  ),
  multiclass = list(
    default = run(X, y_mul),
    accuracy = run(X, y_mul, metric = "accuracy"),
    roc_auc = run(X, y_mul, metric = "roc_auc")
  ),
  spec_fixture = list(default = run(Xs, sf$y, resamples = sf$folds)),
  spec_fixture_collinear = list(default = run(Xsc, sf$y, resamples = sf$folds))
)

# ------------------------------------------------------------ learner failures
marker <- im$marker
y_marker <- im$y_marker

# OLS that refuses to fit unless the marker value is among the training targets.
fails_without_marker <- sharedimp_learner(
  fit = function(x, y) {
    if (!(marker %in% y)) stop("marker row is not in the training set")
    ols$fit(x, y)
  },
  predict = function(model, x) ols$predict(model, x),
  label = "OLS failing without the marker"
)
# OLS whose predictions are NaN unless the marker value was among the training targets.
nan_without_marker <- sharedimp_learner(
  fit = function(x, y) list(beta = ols$fit(x, y), ok = marker %in% y),
  predict = function(model, x) {
    pred <- ols$predict(model$beta, x)
    if (!model$ok) pred[] <- NaN
    pred
  },
  label = "OLS predicting NaN without the marker"
)
# Class frequencies of the training targets, whatever the features.
y_single <- factor(im$y_single_class, levels = c("a", "b"))
prior_learner <- sharedimp_learner(
  fit = function(x, y) as.numeric(table(factor(y, levels = levels(y_single)))) / length(y),
  predict = function(model, x) {
    matrix(model, nrow = nrow(x), ncol = 2L, byrow = TRUE, dimnames = list(NULL, levels(y_single)))
  },
  label = "class frequencies"
)

failures <- list(
  fit_error = run(X, y_marker, metric = "mse", on_error = "warn", learner = fails_without_marker),
  non_finite_predictions = run(X, y_marker, metric = "mse", on_error = "warn", learner = nan_without_marker),
  single_class = run(X, y_single, on_error = "warn", learner = prior_learner)
)
aborts <- function(expr) inherits(tryCatch(expr, error = function(e) e), "error")
failures$on_error_error_aborts <- list(
  fit_error = aborts(run(X, y_marker, metric = "mse", learner = fails_without_marker)),
  non_finite_predictions = aborts(run(X, y_marker, metric = "mse", learner = nan_without_marker)),
  single_class = aborts(run(X, y_single, learner = prior_learner))
)

out <- list(
  meta = list(
    generated = format(Sys.Date()),
    r_version = R.version.string,
    yardstick = as.character(packageVersion("yardstick")),
    nnet = as.character(packageVersion("nnet")),
    sharedimp_r_version = as.character(packageVersion("sharedimp")),
    sharedimp_r_commit = args[[4L]],
    representation_split = "training rows: folds != 1; assessment rows: folds == 1",
    learner = "built-in \"linear\" (lm.fit / glm.fit / nnet::multinom) unless stated otherwise",
    split_labels = "1-based, as reported by R"
  ),
  metrics = metrics,
  representations = list(
    regular = representation_reference(X),
    degenerate = representation_reference(Xd),
    constant = representation_reference(Xc)
  ),
  null_model = null_model,
  importance = importance,
  failures = failures
)
writeLines(
  jsonlite::toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"),
  args[[2L]]
)
