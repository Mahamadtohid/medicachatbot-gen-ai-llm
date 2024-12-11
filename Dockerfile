
FROM python:3.8 

WORKDIR /Chat_bot

COPY requirements.txt /Chat_bot/

RUN pip install -r requirements.txt 

COPY . .

# Expose the port that the app will run on
EXPOSE 8081

# Command to run the application with gunicorn
CMD ["python", "app.py", "--host=0.0.0.0", "--port=8081"]




