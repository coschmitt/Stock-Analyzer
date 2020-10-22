from django import forms

class SearchForm(forms.Form):
    search_box = forms.CharField(widget=forms.TextInput(attrs={'class':"form-control mr-sm-2"}))