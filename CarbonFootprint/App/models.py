from django.db import models

# Create your models here.
class Message(models.Model):
    sender_choices=[
        ("user","User"),
        ("ai","AI")
    ]
    sender=models.CharField(max_length=30,choices=sender_choices)
    # message_user=models.CharField(max_length=30)

    message=models.TextField()
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender} -- {self.message}"
