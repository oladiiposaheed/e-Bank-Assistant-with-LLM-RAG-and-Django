from django.contrib import admin
from web_app.models import QueryHistory
# Register your models here.

@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    
    list_display = ['question', 'created_at']
    search_fields = ['question', 'answer']
    
    