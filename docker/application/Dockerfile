# This dockerfile is complicated.  It has lots of targets, many of which
# are build targets.  They are there so that the ~0.5GB of code needed
# for building (including python pip) doesn't have to be included in the
# final image.
#
# The docker-compose.yaml files in tests and devshell each do a few
# different builds from this Dockerfile.
#
# There are several different "final" versions of the image:
#   included_code : with the Seechange code included, for production running
#   bindmount_code : SeeChange code not included, must be bindmounted, for development (and maybe production)
#   webap : The image that runs on the web ap.  NOTE: you must customize local_overrides.yaml for your case!
#   test_bindmount : dev image including extra stuff for testing, must bindmount SeeChange
#   test_included : dev image including extra stuff for testsing, SeeChange code embedded.
#
# The dev (test) environments are made separately because they bloat the docker image a bit.
#
# The targets in this Dockerfile
#
# base
#   build
#     build_dev
#     included_code_build
#       webap_build
#   bindmount_code      ** One of these two is the primary
#   included_code       **   production image
#   webap
#   base_dev
#     test_included
#     test_bindmount    ** Image used in github CI tests
#
# As of 2026-06-02, the sizes of images from the various stages are as follows:
#   base                0.82GB
#   build               3.50GB
#   build_dev           3.62GB
#   included_code_build 3.63GB
#   webap_build         3.64GB
#   bindmount_code      2.60GB
#   included_code       2.60GB
#   webap               2.61GB
#   base_dev            1.21GB
#   test_included       3.07GB
#   test_bindmount      3.07GB
#
# The "smallest usable image" is bindmount_code, as it includes all the
# pip requirements for production, but doesn't include stuff needed
# for the web server or for tests.
#
# (Note that installing chromium instead of firefox into base_dev
# changes the image size by only 20MB.)

# ======================================================================
# The base target defines the base system that all other images
#   will be built on top of.

FROM debian:trixie-20260518 AS base

LABEL maintainer="Rob Knop <raknop@lbl.gov>"

SHELL ["/bin/bash", "-c"]

# Note: the purge of some pythons below is because there's a
#  version conflict with stuff installed later as a dependency by pip3.
#  (Gotta love duelling package managers.)

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade && \
    DEBIAN_FRONTEND="nonintearctive" TZ="UTC" apt-get -y install -y --no-install-recommends \
      lsb-release \
      curl \
      emacs-nox \
      file \
      fitsverify \
      flex \
      git \
      imagemagick \
      libbz2-1.0 \
      libcfitsio-bin \
      libfftw3-bin \
      libgsl28 \
      liblapack3 \
      less \
      locales \
      missfits \
      netcat-openbsd \
      netpbm \
      postgresql-client \
      procps \
      psfex \
      python3 \
      scamp \
      source-extractor \
      swarp \
      tzdata \
      unzip \
      vim-nox \
      wcslib-dev \
      wcslib-tools \
      wcstools \
      wget  \
    && apt-get -y purge python3-cffi-backend python3-requests python3-dateutil && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# We want to be able to just run python

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Make the sanitized Debian install of sextractor compatible with standard sextractor
RUN ln -sf /usr/bin/source-extractor /usr/bin/sex

# Also fix Debian's ideosyncratic capitalization of swarp
RUN ln -sf /usr/bin/SWarp /usr/bin/swarp

# Create a home directory so that packages can create
# astromatic files if needed.
RUN mkdir /home/seechange
RUN mkdir /home/seechange/.astropy
ENV HOME=/home/seechange
RUN chmod -R 777 /home/seechange

# Generate locale
RUN cat /etc/locale.gen | perl -pe 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' > /etc/locale.gen.new \
    && mv /etc/locale.gen.new /etc/locale.gen
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Just in case we're writing to a archive directory directly,
# make sure directories we create will be group-writeable.
# This is the use case needed for LS4 on Perlmutter.
RUN umask 0002

# Python stuff, trying to subvert injected stuff into containers?
ENV PYTHONSTARTUP=""
ENV PYTHONNOUSERSITE="1"
ENV PYTHONUSERBASE="/tmp"
ENV PYTHONPATH=""

# Some final setups for sanity
ENV LESS=-XLRi

# ======================================================================
# Use a multistage Docker file so that we can install the packages
#   needed for compilation and the like while building things,
#   but don't include those packages in the final Docker image.
#   This saves about ~0.5GB from the final Docker image.
#   (Alas, the pip requirements eat up >1GB, and there's not a lot we
#   can do about that because a big fraction of that is stuff like
#   scipy, astropy, and espeically pytorch (650MB all by itself).)

FROM base AS build

RUN apt-get update && \
    DEBIAN_FRONTEND="nonintearctive" TZ="UTC" apt-get -y install -y --no-install-recommends \
      autoconf \
      automake \
      build-essential \
      cmake \
      gdb \
      gfortran \
      git \
      libbz2-dev \
      libcairo2-dev \
      libcfitsio-dev \
      libcurl4-openssl-dev \
      libfftw3-dev \
      libgsl-dev \
      libjpeg62-turbo-dev \
      liblapacke-dev \
      libnetpbm-dev \
      libopenblas-openmp-dev \
      libpq-dev \
      libssl-dev \
      libtool \
      m4 \
      pkg-config \
      python3-ipykernel \
      python3-pip \
      python3-venv \
      python3-dev \
      swig \
    && apt-get -y purge python3-cffi-backend python3-requests python3-dateutil && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#      libatlas-base-dev \

RUN mkdir /usr/src/seechange
WORKDIR /usr/src/seechange

# Copy all patch files to current working directory
RUN mkdir ./rules
ADD docker/application/patches/patch_* ./rules/

## NOT INCLUDING MPI FOR NOW SINCE ITS HUGE AND WE DON'T CURRENTLY USE IT
## TODO : if we ever uncomment this, need to edit the other stages
##   below to properly copy all the mpich stuff
#
# Need to install mpich here rather than via package manager to ensure
# ABI compatibility.
# ARG mpich_version=4.0.2
# ARG mpich_prefix=mpich-$mpich_version
# RUN curl -L https://www.mpich.org/static/downloads/$mpich_version/$mpich_prefix.tar.gz -O \
#     && tar xf $mpich_prefix.tar.gz \
#     && cd $mpich_prefix \
#     && ./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch \
#     && make -j 16 \
#     && make install \
#     && make clean \
#     && cd .. \
#     && rm -rf $mpich_prefix $mpich_prefix.tar.gz

# Hotpants Alard/Lupton image subtraction
RUN git clone https://github.com/acbecker/hotpants.git \
&& cd hotpants \
    && patch < /usr/src/seechange/rules/patch_hotpants \
    && make -j8 CFITSIOINCDIR=/usr/include CFITSIOLIBDIR=/usr/lib \
    && cp hotpants /usr/bin \
    && cd .. \
    && rm -rf hotpants

RUN mkdir /venv
RUN python3 -mvenv /venv

COPY requirements.txt /seechange/requirements.txt

RUN source /venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install  -r /seechange/requirements.txt


# Pytorch
# Need uv specifically for torch
RUN source /venv/bin/activate \
   && pip install "uv>=0.9.22,<0.10" \
   && uv pip install torch==2.9.1+cpu --find-links https://download.pytorch.org/whl/torch \
   && pip uninstall -y uv


# RBbot
# Remove the requirements.txt file because it will install
# a GPU version of pytorch, and we've already installed
# a CPU version above.
COPY extern/RBbot_inference /usr/src/seechange/RBbot_inference
RUN cd RBbot_inference \
    && rm -f requirements.txt \
    && touch requirements.txt \
    && source /venv/bin/activate \
    && pip install . \
    && cd .. \
    && rm -rf RBbot_inference

# astrometry.net
ARG astrometry_net_ver=0.98
RUN curl -L https://astrometry.net/downloads/astrometry.net-$astrometry_net_ver.tar.gz -O \
   && tar xf astrometry.net-$astrometry_net_ver.tar.gz \
   && cd astrometry.net-$astrometry_net_ver \
   && source /venv/bin/activate \
   && make \
   && make py \
   && make extra \
   && mkdir /astrometry.net \
   && make install INSTALL_DIR=/astrometry.net \
   && cd .. \
   && rm -rf astrometry.net-$astrometry_net_ver


# ======================================================================
# Add the packages needed for tests

FROM build AS build_dev

COPY requirements-dev.txt /seechange/requirements-dev.txt
RUN source /venv/bin/activate \
    && pip install -r /seechange/requirements-dev.txt

# ======================================================================
# This target is for running the "make install" to install
#   the SeeChange code inside the image.  It needs to build
#   off of the build target so all the autotools stuff is
#   there, but we don't want the autotools stuff
#   in the final image.
#
# Right now, this reruns every time you touch a file in this
#  directory.  Could be made faster by moving the data
#  and cache directories out of this tree, though that might
#  require thinking about directory config for tests.  (The
#  goal of this would be to make the ADD . have less to import.)
# (Acutally, it seems to rerun every time...?  Thought required.)

FROM build AS included_code_build

RUN mkdir -p /usr/src/seechange/seechange

# Need to make sure that everything necessary for the SeeChange make
#  install is loaded here.  Don't just ADD . because every time any file
#  anywhere in the tree got touched it would trigger a rebuild of this
#  step; also, that is fairly slow because there may be a lot of
#  gratuitous cached test data in the tree that we really don't need
#  docker to be thinking about.  (This step will still get redone a lot
#  anyway because we will regularly be editing things in the improc,
#  models, pipeline, and util subdirectories.  Fortunately, it should be
#  a pretty fast step.)  (The one is the longest and most wasteful, and
#  is just there so that the creation of util/githash.py works in
#  util/Makefile.am.)
ADD .git /usr/src/seechange/seechange/.git
ADD configure.ac /usr/src/seechange/seechange/configure.ac
ADD Makefile.am /usr/src/seechange/seechange/Makefile.am
ADD requirements.txt /usr/src/seechange/seechange/requirements.txt
ADD improc/ /usr/src/seechange/seechange/improc/
ADD models/ /usr/src/seechange/seechange/models/
ADD pipeline/ /usr/src/seechange/seechange/pipeline/
ADD util/ /usr/src/seechange/seechange/util/
ADD share/ /usr/src/seechange/seechange/share/
# archive.py and rkauth_client.py need special treatment because they're symbolic links in the repo
ADD util/archive.py /usr/src/seechange/seechange/util/archive.py
ADD util/rkauth_client.py /usr/src/seechange/seechange/util/rkauth_client.py
# special treatment for share/RBbot_inference/models because of sybmolic links
ADD share/RBbot_models/* /usr/src/seechange/seechange/share/RBbot_models/

WORKDIR /usr/src/seechange/seechange
RUN autoreconf --install
RUN ./configure --with-installdir=/seechange/lib
RUN make install

# ======================================================================
# This is the target used to do a make install for the webap's
#   specific server and client code.

FROM included_code_build AS webap_build

COPY requirements-webap.txt /seechange/requirements-webap.txt
RUN source /venv/bin/activate \
    && pip install -r /seechange/requirements-webap.txt

RUN mkdir -p /usr/src/seechange/webap
ADD webap/ /usr/src/seechange/webap/
ADD extern/rkwebutil /usr/src/seechange/extern/rkwebutil

WORKDIR /usr/src/seechange/webap
RUN make INSTALLDIR=/webap_code install

# ======================================================================
# This target is for an image that will not include any
#  of the SeeChange code inside the image, but will need
#  to bind-mount it from the host.

FROM base AS bindmount_code

COPY --from=build /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build /venv/ /venv/
COPY --from=build /astrometry.net/ /astrometry.net/
ENV PATH=/venv/bin:$PATH

ENV PYTHONPATH="/seechange"
WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]

# =====================================================================
# This is the target for an image that includes the SeeChange code.

FROM base AS included_code

COPY --from=build /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build /venv/ /venv/
COPY --from=build /astrometry.net/ /astrometry.net/
COPY --from=included_code_build /seechange/lib /seechange/lib

COPY default_config.yaml /seechange/default_config.yaml
COPY pipeline_conifg.yaml /seechange/pipeline_config.yaml
COPY site_config.yaml /seechange/site_config.yaml
COPY default_config_extragalactic.yaml /seechange/default_config_extragalactic.yaml
COPY default_config_galactic.yaml /seeconage/default_config_galactic.yaml
COPY pipeline_config_galactic.yaml /seechange/pipeline_config_galactic.yaml
COPY docker/application/local_overrides.yaml /seechange/local_overrides.yaml

ENV PATH=/venv/bin:$PATH
ENV SEECHANGE_CONFIG=/seechange/default_config.yaml
ENV PYTHONPATH="/seechange/lib"

RUN mkdir /seechange/data
RUN mkdir /seechange/temp

WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]


# ======================================================================
# This is the target for the webap

FROM base AS webap

COPY --from=webap_build /usr/bin/hotpants /usr/bin/hotpants
COPY --from=webap_build /venv/ /venv/
COPY --from=webap_build /astrometry.net/ /astrometry.net/
COPY --from=webap_build /webap_code/ /webap_code/
COPY --from=webap_build /seechange/lib /seechange/lib

COPY default_config.yaml /seechange/default_config.yaml
COPY pipeline_conifg.yaml /seechange/pipeline_config.yaml
COPY site_config.yaml /seechange/site_config.yaml
COPY default_config_extragalactic.yaml /seechange/default_config_extragalactic.yaml
COPY default_config_galactic.yaml /seeconage/default_config_galactic.yaml
COPY pipeline_config_galactic.yaml /seechange/pipeline_config_galactic.yaml
COPY webap/seechange_webap.yaml /seechange/seechange_webap.yaml
COPY webap/webap_local_overrides.yaml /seechange/webap_local_overrides.yaml
COPY webap/webap_local_augments.yaml /seechange/webap_local_augments.yaml

# Copy in the self-signed certificate we need to use in tests.
# This should *not* be used in production, even though it's
# sitting there in the image.
COPY docker/application/webap_bogus_key.pem /webap_code/bogus_key.pem
COPY docker/application/webap_bogus_cert.pem /webap_code/bogus_cert.pem
RUN chmod a+r /webap_code/bogus*pem

ENV PATH=/venv/bin:$PATH
ENV SEECHANGE_CONFIG=/seechange/seechange_webap.yaml
ENV PYTHONPATH="/seechange/lib"

RUN mkdir /sessions
RUN chmod 777 /sessions

WORKDIR /webap_code/
RUN chmod 755 run_webap.sh

CMD [ "./run_webap.sh", "8080" ]

# ======================================================================
# This is the base image plus some dev stuff (including firefox...!)

FROM base AS base_dev

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y install --no-install-recommends \
       firefox-esr iproute2 net-tools rlwrap socat \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ======================================================================
# This is the image that includes a web browser and includes the code
# built into the docker image.  This is what's used in our automatically
# run tests.  (...actually, I think it may use test_bindmount? hmmm...)

FROM base_dev AS test_included

COPY --from=build_dev /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build_dev /venv/ /venv/
COPY --from=build_dev /astrometry.net/ /astrometry.net/
COPY --from=included_code_build /seechange/lib /seechange/lib

COPY default_config.yaml /seechange/default_config.yaml
COPY pipeline_conifg.yaml /seechange/pipeline_config.yaml
COPY site_config.yaml /seechange/site_config.yaml
COPY default_config_extragalactic.yaml /seechange/default_config_extragalactic.yaml
COPY default_config_galactic.yaml /seeconage/default_config_galactic.yaml
COPY pipeline_config_galactic.yaml /seechange/pipeline_config_galactic.yaml
COPY docker/application/local_overrides.yaml /seechange/local_overrides.yaml

ENV PATH=/venv/bin:$PATH
ENV SEECHANGE_CONFIG=/seechange/default_config.yaml
ENV PYTHONPATH="/seechange/lib"

RUN mkdir /seechange/data
RUN mkdir /seechange/temp

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]


# ======================================================================
# This is the image that includes a web browser but expects the code
# to be bind-mounted at /seechange.  This is for dev purposes.

FROM base_dev AS test_bindmount

COPY --from=build_dev /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build_dev /venv/ /venv/
COPY --from=build_dev /astrometry.net/ /astrometry.net/

ENV PATH=/venv/bin:$PATH
ENV PYTHONPATH="/seechange"

WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]
