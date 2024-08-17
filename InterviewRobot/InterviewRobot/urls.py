"""InterviewRobot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
'''
urls.py
'''
from django.contrib import admin
from django.urls import path
import sys
sys.path.append('./')
from run import InterviewRobot,pdf2txt_server,key_word_show,get_question,send_answer


urlpatterns = [
    path('InterviewRobot', InterviewRobot),
    path('pdf2txt/', pdf2txt_server),
    path('key_word_show/', key_word_show),
    path('get_question/', get_question),
    path('send_answer/',send_answer),
]
