# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# copy all the files to the container
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN chmod +x start.sh

CMD ["./start.sh"]

# run the command
#CMD ["python", "./main.py", ""]
#CMD "/usr/local/bin/python ./main.py --range 0-5"
