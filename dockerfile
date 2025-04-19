FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app/main.py
ENV FLASK_ENV=development

EXPOSE 5001

CMD ["python", "-m", "app.main"]
