FROM python:3.8
COPY requirements.txt /dashboard/
RUN pip install --upgrade pip
RUN pip install -r /dashboard/requirements.txt
COPY . /dashboard/
WORKDIR /dashboard/
EXPOSE 8501
CMD ["sh", "run.sh"]