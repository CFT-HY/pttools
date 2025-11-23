# The standard Debian image is used instead of the Alpine one,
# as Alpine uses a different version of the C standard library,
# and therefore the usual Python wheels don't work on Alpine.
FROM python:3.13

# Install generic dependencies
RUN apt-get update \
    && apt-get install -y cmake gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade --root-user-action=ignore --no-cache-dir build pip wheel

# Install project dependencies
COPY requirements.txt /pttools/
RUN pip install --root-user-action=ignore --no-cache-dir -r /pttools/requirements.txt

# Build the project
COPY pyproject.toml /pttools/
COPY pttools pttools/pttools
RUN cd /pttools \
    && ls \
    && python -m build \
    && pip install --root-user-action=ignore --no-cache-dir /pttools/dist/pttools*.whl \
    && cd / \
    && rm -r /pttools \
