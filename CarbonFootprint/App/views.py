from django.shortcuts import render,redirect,HttpResponse
import  google.generativeai as genai 
from .models import Message
import torch
# Create your views here.
def index(request):
    return render(request,'index.html')
def llm(prompt):
    system_prompt=f"""You are a helpful and friendly assistant that analyzes users' activity data (e.g., food, travel, electricity, shopping) to estimate carbon emissions. Provide personalized insights and actionable suggestions to help reduce their environmental impact. Use simple language, metric units (g/kg CO₂), and explain concepts clearly. Ask follow-up questions if data is missing or unclear. 
    
    User Input : {prompt}
    """
    api_key="AIzaSyDHPFE8jF2ZtCUN8G6arkp7FEEl6S_C8CQ"
    genai.configure(api_key=api_key)
    model=genai.GenerativeModel(model_name='gemini-2.0-flash')
    response=model.generate_content(system_prompt)
    # chat = model.start_chat()
    # response = chat.send_message(prompt)
    return response.text

def chat_view(request):
    if request.method=="POST":
        user_input=request.POST.get("user_input")
        if user_input:
            Message.objects.create(sender=request.user,message=user_input)

            ai_response=llm(prompt=user_input)
            Message.objects.create(sender='AI',message=ai_response)
            return redirect("chatroom")
        

    room_msgs=Message.objects.order_by("created_at")




    return render(request,'chat.html',{"messages":room_msgs})



from django.contrib.auth.models import User
from django.contrib.auth import login,logout,authenticate
def loginUser(request):
    # if request.user.is_authenticated():
    #     return redirect("home")
    if request.method=="POST":
        username=request.POST.get("username")
        password=request.POST.get("password")
        user=authenticate(request,username=username,password=password)
        if user is None:
            login(request,user)
            return redirect('main')
    return render(request,"login.html")



def signupUser(request):
    if request.method=="POST":
        username=request.POST.get("name")
        email=request.POST.get("email")
        password=request.POST.get("password")

        
        if User.objects.filter(username=username).exists():
            return HttpResponse("User already exists")
        else:


            user=User.objects.create(username=username,email=email,password=password)
            user.save()
            login(request,user)
            return redirect("main")
    return render(request,'signup.html')


def logout(request):
    logout(request)
    return redirect('login')

def main(request):
    return render(request,"main.html")


def predict(request):
    return render(request,'form.html')

from torch import nn
class CarbonPredModel(nn.Module):
  def __init__(self,in_channel,out):
    super().__init__()
    self.feedF=nn.Sequential(
        nn.Linear(in_features=in_channel,out_features=2*in_channel),
        nn.ReLU(),
        nn.Linear(in_features=2*in_channel,out_features=2*in_channel),
        nn.ReLU(),
        nn.Linear(in_features=2*in_channel,out_features=2*in_channel),
        nn.ReLU(),
        nn.Linear(in_features=2*in_channel,out_features=out)
    )
  def forward(self,x):
    return self.feedF(x)


model = CarbonPredModel(12,1)

# Load the weights
import os
import torch

# Get the full path to the model file, relative to the current file
import os

# Correct the path by pointing to the new location of the .pth file
model_path = os.path.join(os.path.dirname(__file__), 'carbon_prediction_model.pth')
print(f"Model path: {model_path}")  # Optional: for debugging the path

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()  
def make_predict(request):
    if request.method == "POST":
        # 1. Extract form data
        transport_mode = request.POST.get("transport_mode")
        fuel_type = request.POST.get("fuel_type")
        distance_km_weekly = float(request.POST.get("distance_km_weekly", 0))
        diet_type = request.POST.get("diet_type")
        meat_kg = float(request.POST.get("meat_consumption_kg_per_week", 0))
        electricity_kwh = float(request.POST.get("monthly_electricity_kwh", 0))
        energy_source = request.POST.get("energy_source")
        clothes = int(request.POST.get("clothing_items_bought_per_month", 0))
        electronics = int(request.POST.get("electronics_purchased_yearly", 0))
        plastic = float(request.POST.get("plastic_waste_kg_per_week", 0))
        household_size = int(request.POST.get("household_size", 1))
        uses_solar = request.POST.get("uses_solar_panels") == "True"

        # 2. Preprocess inputs (map categories to numbers as your model expects)
        transport_dict = {"Bus": 0, "Bike": 1, "Car": 2, "Flight": 3, "Train": 4}
        fuel_dict = {"Diesel": 0, "Petrol": 1, "Electric": 2, "Hybrid": 3}
        diet_dict = {"Vegan": 0, "Vegetarian": 1, "Meat-heavy": 2}
        energy_dict = {"Solar": 0, "Wind": 1, "Coal": 2}  # Add more if needed

        input_vector = [
            transport_dict.get(transport_mode, 0),
            fuel_dict.get(fuel_type, 0),
            distance_km_weekly,
            diet_dict.get(diet_type, 0),
            meat_kg,
            electricity_kwh,
            energy_dict.get(energy_source, 0),
            clothes,
            electronics,
            plastic,
            household_size,
            1 if uses_solar else 0,
        ]

        input_tensor = torch.tensor([input_vector], dtype=torch.float32)

        # 3. Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # 4. Render result
        return HttpResponse(f"<h2>Your predicted emission of carbon footprint for this week is: {round(prediction, 5)} kg</h2>")

    return render(request, "form.html")

def delete(request):
    Message.objects.all().delete()
    return redirect('chatroom')