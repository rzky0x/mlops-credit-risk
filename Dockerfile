FROM tensorflow/serving:2.15.1

# Copy the serving model (latest version) into the container
COPY ./rzky0x-pipeline/serving_model /models/credit-risk-model

# Copy monitoring configuration
COPY ./monitoring/prometheus.config /models/monitoring.config

# Set environment variables
ENV MODEL_NAME=credit-risk-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8501

# Expose REST API port
EXPOSE 8501

# Use shell form so $PORT is expanded at runtime (Railway injects PORT)
CMD tensorflow_model_server \
    --port=8500 \
    --rest_api_port=${PORT} \
    --model_name=credit-risk-model \
    --model_base_path=/models/credit-risk-model \
    --monitoring_config_file=/models/monitoring.config
