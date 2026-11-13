# syntax=docker/dockerfile:1

# sharedimp container image.
#
#   docker build -t sharedimp-py .
#   docker run --rm sharedimp-py python /opt/sharedimp/smoke_test.py
#
# Two stages: the wheel is built from the sources in a throw-away stage, and
# the final image contains only the installed package and its runtime
# dependencies (no sources, tests, docs, build tools or pip cache).

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- build stage
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

# -------------------------------------------------------------- runtime stage
FROM python:${PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install sharedimp and its runtime dependencies (numpy, scipy, pandas,
# scikit-learn, joblib) from the wheel built above.
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    python -m pip install /wheels/sharedimp-*.whl

COPY docker/smoke_test.py /opt/sharedimp/smoke_test.py

# Unprivileged default user with a writable working directory.
RUN useradd --create-home --uid 1000 sharedimp \
    && install -d -o sharedimp -g sharedimp /work
USER sharedimp
WORKDIR /work

# Build-time check: import the package and run a small shared_importance()
# regression example with scikit-learn. The build fails if it fails.
RUN python /opt/sharedimp/smoke_test.py

CMD ["python"]
