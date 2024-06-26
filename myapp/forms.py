from django import forms
from .models import Image

class ImageForm(forms.ModelForm):
 additional_field = forms.CharField(label='PassPort ID', max_length=100)

 class Meta:
  model = Image
  fields = '__all__'
  exclude = ['unique_id','name', 'surname']
  labels = {'photo':''}

  def __init__(self, *args, **kwargs):
   super().__init__(*args, **kwargs)

   # Use TextInput widget for the photo field
   self.fields['photo'].widget = forms.TextInput(attrs={'type': 'text'})