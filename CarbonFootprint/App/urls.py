from django.contrib import admin
from django.urls import path,include
from . import views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chatroom/',views.chat_view,name='chatroom'),
    path('login/',views.loginUser,name='login'),
    path('signup/',views.signupUser,name='signup'),
    path('main/',views.main,name='main'),
    path('predict_form/',views.predict,name='predict_form'),
    path('predict/', views.make_predict, name='make_predict'),
    path('delete/',views.delete,name='delete'),
    



]