# syntax=docker/dockerfile:1
# check=error=true
# https://docs.docker.com/build/checks/#fail-build-on-check-violations

# The CUDA image is used to provide GPU compatibility for software that depends on PTtools.
# Ubuntu 26.04 includes Python 3.14.
# The image should not be based on Alpine, as it uses a different version of the C standard library,
# and therefore the usual Python wheels don't work on Alpine.
ARG CUDA_IMAGE="nvidia/cuda:13.3.1-base-ubuntu26.04"

# The Python libraries are installed in a virtual environment,
# as the system Python of Ubuntu is externally managed (PEP 668).
# The virtual environment is a self-contained folder,
# and therefore it can be copied to the final image without the compilers that were used for building it.
ARG VENV_PATH="/opt/venv"


# ---
# Builder stage: compiles the dependencies that don't have wheels for all platforms
# ---
FROM ${CUDA_IMAGE} AS builder

# The architecture is used for separating the build caches of the different platforms.
ARG TARGETARCH
ARG VENV_PATH

# Setting these environment variables is equivalent to activating the virtual environment.
ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

# Install generic dependencies.
# The compilers are needed for the dependencies that don't have wheels for all platforms, such as NumbaLSODA.
# The automatic cleanup is disabled to preserve the apt caches.
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake gfortran patchelf python3-dev python3-venv

# Install project dependencies
COPY requirements.txt /opt/
RUN \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=pip-${TARGETARCH} \
    python3 -m venv "${VENV_PATH}" \
    && pip install -r /opt/requirements.txt \
    # NumbaLSODA requires an executable stack, but glibc >= 2.41 refuses to enable it when loading a shared library. \
    # https://github.com/Nicholaswogan/numbalsoda/issues/34 \
    && patchelf --clear-execstack "${VENV_PATH}"/lib/python*/site-packages/numbalsoda/*.so

# Build and install the project.
# The bind mount has to be writable, as setuptools writes the .egg-info to the source directory.
RUN \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=pip-${TARGETARCH} \
    --mount=type=bind,source=.,target=/src,rw \
    pip install /src


# ---
# Final stage: contains only CUDA, Python and the virtual environment
# ---
FROM ${CUDA_IMAGE}

ARG TARGETARCH
ARG VENV_PATH

ENV VIRTUAL_ENV="${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV LD_LIBRARY_PATH="${VENV_PATH}/lib:${LD_LIBRARY_PATH}"

# Python has to be installed, as the virtual environment links to the system interpreter.
# Libgfortran is the runtime library of gfortran, which is needed by the compiled dependencies such as NumbaLSODA.
# Libgomp is the OpenMP runtime of GCC, which is used by the OpenMP threading layer of Numba.
# If you need to compile additional Python libraries in an image that is based on this one,
# then install the compilers with: apt-get install build-essential cmake gfortran python3-dev
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib-${TARGETARCH} \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgfortran5 libgomp1 python3

COPY --from=builder ${VENV_PATH} ${VENV_PATH}
