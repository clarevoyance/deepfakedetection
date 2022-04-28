FROM python:3.8

WORKDIR /app

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /app

CMD streamlit run --server.port 8080 --server.enableCORS false app.py