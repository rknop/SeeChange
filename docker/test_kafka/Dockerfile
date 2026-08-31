FROM apache/kafka:4.2.0

USER root

# Install netcat for the docker compose health check
RUN apk update \
  && apk add netcat-openbsd

USER appuser
