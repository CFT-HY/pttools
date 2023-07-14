# The standard Debian image is used instead of the Alpine one,
# as Alpine uses a different version of the C standard library,
# and therefore the usual Python wheels don't work on Alpine.
FROM python:3.11

RUN apt-get update \
    && apt-get install -y cmake gfortran \
    && apt-get autoremove \
    && apt-get clean \
    && pip install --no-cache-dir --upgrade build
COPY requirements.txt /pttools/
RUN pip install --no-cache-dir -r /pttools/requirements.txt
COPY pyproject.toml setup.cfg /pttools/
COPY pttools pttools/pttools
RUN cd /pttools \
    && ls \
    && python -m build \
    && pip install --no-cache-dir /pttools/dist/pttools*.whl \
    && cd / \
    && rm -r /pttools \
