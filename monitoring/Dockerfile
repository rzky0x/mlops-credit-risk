FROM prom/prometheus:latest

# Copy Prometheus configuration
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose Prometheus web UI port
EXPOSE 9090

# Default entrypoint from base image handles starting Prometheus
