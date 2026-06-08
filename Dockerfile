# 1. Start with Python
FROM python:3.10-slim


# 2. Set the work directory
WORKDIR /app

RUN pip install --upgrade pip

# 3. Download libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy everything from your folder to the container
COPY . .

# 5. Make the script "runnable"
RUN chmod +x run.sh

# 6. Tell Docker what to do when it starts
CMD ["bash", "run.sh"]