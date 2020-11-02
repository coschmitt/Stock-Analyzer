import requests
import json
import pprint
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd

from django.http import HttpResponseRedirect
from django.db.models import ObjectDoesNotExist
from django.shortcuts import render
from django.contrib.auth import login, authenticate

from django.views.generic.edit import FormView
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.models import User

from StockAnalyzer.forms import SearchForm

# API Key: bu8fm1v48v6qo2tha7gg
yf.pdr_override()

# Create your views here.
def index(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect("/stock-analyzer/auth_home")
    else:       # Unregistered User
        return HttpResponseRedirect("/stock-analyzer/sign-in")


class SignUp(FormView):
    success_url = "/stock-analyzer/auth_home"
    form_class = UserCreationForm
    template_name = "StockAnalyzer/signup.html"

    def form_valid(self, form):
        form.save()
        user = User.objects.get(username=form.data.get('username'))
        login(self.request, user)
        form.save()

        return HttpResponseRedirect("/stock-analyzer/auth_home")


class SignOut(LogoutView):
    template_name = 'registration/logged_out.html'
    success_url_allowed_hosts = "/stock-analyzer/sign-in"


def search(request):
    if request.method == 'POST':
        ticker = request.POST.get('search_box')
        # r = requests.get('https://finnhub.io/api/v1/stock/metric?symbol='+ ticker +'&token=bu8fm1v48v6qo2tha7gg')
        # r = requests.get('https://finnhub.io/api/v1/stock/profile2?symbol=' + ticker + '&token=bu8fm1v48v6qo2tha7gg')
        # r_json = r.json()
        data = pdr.get_data_yahoo(
            str(ticker),
            start = pd.to_datetime('2017-01-01'),
            end = '2019-01-29'
        )
        print(data)


        # pprint.pprint(r_json)


    return HttpResponseRedirect("/stock-analyzer/auth_home")


class SignIn(LoginView):
    success_url = "/stock-analyzer/auth_home"
    form_class = AuthenticationForm
    template_name = "StockAnalyzer/signin.html"

    def form_valid(self, form):
        """Security check complete. Log the user in."""
        login(self.request, form.get_user())
        return HttpResponseRedirect("/stock-analyzer/auth_home")


def home(request):
    user = request.user
    if not user.is_authenticated:
        return HttpResponseRedirect("/stock-analyzer/sign-in")
    return render(request, 'StockAnalyzer/home.html', context={
        'user':user,
        'form':SearchForm()
    })


