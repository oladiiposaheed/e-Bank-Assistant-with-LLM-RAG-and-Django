from django.db import models

# Create your models here.

class QueryHistory(models.Model):
    '''Store user queries for reference.'''
    
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f'{self.question}'
    
    