from flask import Flask, Response, render_template, make_response, jsonify, request as req
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import boto3
from datetime import datetime
import requests
import speech_recognition as sr
from scipy.io import wavfile
import os