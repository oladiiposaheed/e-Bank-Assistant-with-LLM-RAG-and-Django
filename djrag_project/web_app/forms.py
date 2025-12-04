from django import forms

class QueryForm(forms.Form):
    '''Simple form for user queries.'''
    
    question = forms.CharField(
        max_length=1000,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Ask about SafeBank services...',
            'row': 3
        })
    )