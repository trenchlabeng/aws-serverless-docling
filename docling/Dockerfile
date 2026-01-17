# Use AWS Lambda Python 3.13 base image (Amazon Linux 2023 minimal)
FROM public.ecr.aws/lambda/python:3.13

# Set the working directory to the Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements file first
COPY docling/requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip uninstall -y opencv-python opencv-python-headless
# RUN pip install --no-cache-dir opencv-python-headless
# Optional: Set Torch cache directory
ENV TORCH_HOME=/tmp/torch
ENV MODULE_PATH=/tmp
ENV HF_HOME=/tmp/huggingface

RUN docling-tools models download-hf-repo sentence-transformers/all-MiniLM-L6-v2
# Copy the rest of the application code
COPY docling/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY docling/docling_parser.py ${LAMBDA_TASK_ROOT}/

# Set the entry point for the Lambda function
CMD ["lambda_function.lambda_handler"]