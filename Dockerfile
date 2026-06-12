FROM tensorflow/serving:2.15.0

# Copy the serving model (latest version) into the container
COPY ./rzky0x-pipeline/serving_model /models/credit-risk-model

# Copy monitoring configuration
COPY ./monitoring/prometheus.config /models/monitoring.config

# Set environment variables
ENV MODEL_NAME=credit-risk-model
ENV MODEL_BASE_PATH=/models

# Expose REST API port
EXPOSE 8501

# Entrypoint — TF Serving with monitoring enabled
# Railway injects PORT env var; TF Serving uses rest_api_port
ENTRYPOINT ["tensorflow_model_server"]
CMD ["--port=8500", \
     "--rest_api_port=8501", \
     "--model_name=credit-risk-model", \
     "--model_base_path=/models/credit-risk-model", \
     "--monitoring_config_file=/models/monitoring.config"]
