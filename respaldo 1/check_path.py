from flask import Flask
import os

app = Flask(__name__)
print(f"Instance path: {app.instance_path}")
print(f"CWD: {os.getcwd()}")
