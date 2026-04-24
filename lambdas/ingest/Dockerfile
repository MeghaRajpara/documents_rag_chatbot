# AWS provides a base image specifically for Lambda + Python
# It includes the Lambda runtime interface client already
FROM public.ecr.aws/lambda/python:3.11

# Copy and install dependencies first
# Docker caches this layer — only rebuilds if requirements.txt changes
# This is best practice for faster builds
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy handler code
COPY handler.py .

# Tell Lambda which function to call
# Format: filename.function_name
CMD ["handler.handler"]