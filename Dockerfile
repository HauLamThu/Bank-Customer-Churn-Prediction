FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# FROM python:3.11 
# WORKDIR /app  
# COPY . .  
# RUN pip install pandas scikit-learn fastapi uvicorn  
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]  