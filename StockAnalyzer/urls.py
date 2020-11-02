from django.contrib import admin
from django.urls import path, include
from . import views
import sys


urlpatterns = [
    path("", views.index, name='home'),
    path("sign-up", views.SignUp.as_view(), name="sign-up"),
    path("sign-in", views.SignIn.as_view(), name="sign-in"),
    path("auth_home", views.home, name="auth_home"),
    path("signout", views.SignOut.as_view(), name='sign-out'),
    path("search", views.search, name='search'),
    # path("save/<str:address>/<str:query>", views.save, name='save'),
    # path("saved-homes", views.list_saved_homes, name='saved-homes'),

]
